import multiprocessing
from os.path import exists, join

import numpy
import progress.bar
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

# Fix multiprocessing on macOS - must use 'spawn' for MPS compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from data import (get_test_set, get_training_set, get_validation_set,
                  worker_init_fn)


class Singleton(type):
    """Thread-safe Singleton metaclass with reset capability for testing and tuning."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
    @classmethod
    def reset(mcs, cls=None):
        """Reset singleton instance(s) to allow fresh initialization.
        
        Args:
            cls: Specific class to reset. If None, resets all singletons.
        
        Usage:
            Singleton.reset(Settings)  # Reset only Settings
            Singleton.reset()          # Reset all singletons
        """
        if cls is not None:
            mcs._instances.pop(cls, None)
        else:
            mcs._instances.clear()


class Settings(metaclass=Singleton):
    def __init__(self):
        with open("settings.yaml", 'r') as stream:
            self.dictionary = yaml.load(stream, Loader=Loader)
        self._input_path = self.dictionary['input_path']
        self._output_path = self.dictionary['output_path']
        self._model_path = self.dictionary['model_path']
        self._upscale_factor = self.dictionary['upscale_factor']
        # Validate upscale_factor
        valid_factors = (2, 3, 4, 8)
        if self._upscale_factor not in valid_factors:
            raise ValueError(
                f"upscale_factor must be one of {valid_factors}, got {self._upscale_factor}"
            )
        self._mode = self.dictionary['mode']
        self._n_epochs = self.dictionary['epochs_number']
        self._epoch = self.dictionary['epoch']
        self._checkpoint_freq = self.dictionary['checkpoint_frequency']
        self._batch_size = self.dictionary['batch_size']
        self._test_batch_size = self.dictionary['test_batch_size']
        self._lr = self.dictionary['learning_rate']
        self._momentum = self.dictionary['momentum']
        self._weight_decay = self.dictionary['weight_decay']
        self._threads = self.dictionary['threads']
        self._optimizer_type = self.dictionary['optimizer']
        self._psnr_delta = self.dictionary['psnr_delta']
        self._stuck_level = self.dictionary['stuck_level']
        self._target_min_psnr = self.dictionary['target_min_psnr']
        self._cuda = self.dictionary['cuda']
        self._mps = self.dictionary.get('mps', False)
        self._tuning = self.dictionary['tuning']
        self._trials = self.dictionary['trials']
        self._mp = self.dictionary['mixed_precision']
        self._compile = self.dictionary.get('compile_model', False)
        self._channels_last = self.dictionary.get('channels_last', False)
        self._persistent_workers = self.dictionary.get('persistent_workers', False)
        self._optimized = self.dictionary['optimized']
        self._num_classes = self.dictionary['num_classes']
        self._separable = self.dictionary['separable']
        self._scheduler_enabled = self.dictionary['scheduler']
        self._pruning = self.dictionary['pruning']
        self._prune_amount = self.dictionary['prune_amount']
        self._preload = self.dictionary['preload']
        self._preload_path = self.dictionary['preload_path']
        self._seed = self.dictionary['seed']
        self._pb = self.dictionary['show_progress_bar']
        self._profiler = self.dictionary['show_profiler']
        self._show_result = self.dictionary['show_result']
        self._cycles = self.dictionary['cycles']
        train_set = get_training_set(self._upscale_factor)
        test_set = get_test_set(self._upscale_factor)
        val_set = get_validation_set(self._upscale_factor)
        if self._optimized:
            from model_ench import ObjectAwareESPCN as espcn
            self._pruning = False
        else:
            from model import ESPCN as espcn
        # Device selection priority: CUDA > MPS (Apple Neural Engine) > CPU
        if self._cuda and torch.cuda.is_available():
            self._device = torch.device("cuda")
            torch.cuda.manual_seed(self._seed)
            torch.backends.cudnn.benchmark = True
        elif self._mps and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            # MPS-specific optimizations
            # Enable memory-efficient attention for Apple Silicon
            if hasattr(torch.backends, 'mps'):
                torch.mps.set_per_process_memory_fraction(0.0)  # Allow dynamic memory allocation
        else:
            self._device = torch.device("cpu")
            # CPU optimizations
            torch.set_num_threads(self._threads)
        torch.manual_seed(self._seed)
        numpy.random.seed(self._seed)
        
        if self._cuda and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        self._model = espcn(self._num_classes, self._upscale_factor, separable = self._separable)
        
        if self._channels_last:
            self._model = self._model.to(memory_format=torch.channels_last)
        
        self._model = self._model.to(self._device)
        if self._preload and exists(self._preload_path):
            self._model = torch.load(self._preload_path, weights_only=False, map_location=self._device)
        
        if self._compile and hasattr(torch, 'compile') and not (self._mps and torch.backends.mps.is_available()):
            try:
                self._model = torch.compile(self._model, backend='inductor', mode='reduce-overhead')
            except Exception as e:
                import logging
                logging.warning(f"torch.compile failed, continuing without compilation: {e}")
        
        self._criterion = nn.MSELoss().to(self._device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        if self._optimizer_type == 'SGD':
            self._optimizer = optim.SGD(self._model.parameters(), lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay)
        # Determine device type for GradScaler (MPS uses CPU scaler as MPS doesn't support native AMP scaler)
        device_type = str(self._device.type)
        scaler_enabled = self._mp and device_type == 'cuda'  # AMP GradScaler only supported on CUDA
        self._scaler = torch.amp.GradScaler(device='cuda' if device_type == 'cuda' else 'cpu', enabled=scaler_enabled)
        # Optimized DataLoader settings
        # pin_memory: Faster CPU->GPU transfer (CUDA only)
        # persistent_workers: Keep workers alive between epochs (reduces spawn overhead)
        # prefetch_factor: Number of batches loaded in advance per worker
        # NOTE: On MPS, use 0 workers to avoid multiprocessing deadlocks
        pin_mem = self._cuda and torch.cuda.is_available()
        effective_threads = 0 if self._mps else self._threads  # MPS: disable multiprocessing
        persistent = self._persistent_workers and effective_threads > 0
        prefetch = 4 if persistent else 2  # Higher prefetch with persistent workers
        
        self._training_data_loader = DataLoader(
            dataset=train_set, 
            num_workers=effective_threads,
            batch_size=self._batch_size, 
            shuffle=True,
            pin_memory=pin_mem,
            persistent_workers=persistent,
            prefetch_factor=prefetch if effective_threads > 0 else None,
            worker_init_fn=worker_init_fn if effective_threads > 0 else None
        )
        self._testing_data_loader = DataLoader(
            dataset=test_set, 
            num_workers=effective_threads,
            batch_size=self._test_batch_size, 
            shuffle=False,
            pin_memory=pin_mem,
            persistent_workers=persistent,
            prefetch_factor=prefetch if effective_threads > 0 else None,
            worker_init_fn=worker_init_fn if effective_threads > 0 else None
        )
        self._validation_data_loader = DataLoader(
            dataset=val_set, 
            num_workers=effective_threads,
            batch_size=self._test_batch_size, 
            shuffle=True,
            pin_memory=pin_mem,
            persistent_workers=persistent,
            prefetch_factor=prefetch if effective_threads > 0 else None,
            worker_init_fn=worker_init_fn if effective_threads > 0 else None
        )
        if self._scheduler_enabled:
            # Ensure optimizer step count is set BEFORE creating scheduler to avoid warning
            self._optimizer._step_count = 1
            self._scheduler = optim.lr_scheduler.OneCycleLR(
                self._optimizer,
                max_lr=self._lr,
                steps_per_epoch=len(self._training_data_loader),
                epochs=self._n_epochs,
                anneal_strategy='cos',
                final_div_factor=1e4
            )
            # Also set the scheduler's internal counter to match
            self._scheduler._step_count = 1

    def update_setting(self, key, value):
        self.dictionary[key] = value
        # with open("settings.yaml", 'r') as stream:
        #     current_settings = yaml.load(stream, Loader=Loader)
        # current_settings[key] = value
        # with open("settings.yaml", 'w') as stream:
        #     yaml.dump(current_settings, stream, default_flow_style=False)

    @property
    def model(self):
        return self._model

    @property
    def input_path(self):
        return self._input_path

    @property
    def output_path(self):
        return self._output_path

    @property
    def model_path(self):
        return self._model_path

    @property
    def upscale_factor(self):
        return self._upscale_factor

    @property
    def mode(self):
        return self._mode

    @property
    def epochs_number(self):
        return self._n_epochs

    @property
    def epoch(self):
        return self._epoch

    @property
    def checkpoint_frequency(self):
        return self._checkpoint_freq

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        self.update_setting('batch_size', self._batch_size)

    @property
    def test_batch_size(self):
        return self._test_batch_size

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, lr: float):
        self._lr = lr
        self.update_setting('learning_rate', lr)

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, momentum: float):
        self._momentum = momentum
        self.update_setting('momentum', momentum)

    @property
    def weight_decay(self):
        return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, weight_decay: float):
        self._weight_decay = weight_decay
        self.update_setting('weight_decay', weight_decay)

    @property
    def threads(self):
        return self._threads

    @property
    def optimizer_type(self):
        return self._optimizer_type

    @optimizer_type.setter
    def optimizer_type(self, optimizer_type):
        self._optimizer_type = optimizer_type
        self.update_setting('optimizer', optimizer_type)

    @property
    def psnr_delta(self):
        return self._psnr_delta

    @property
    def stuck_level(self):
        return self._stuck_level

    @property
    def target_min_psnr(self):
        return self._target_min_psnr

    @property
    def cuda(self):
        return self._cuda

    @property
    def mps(self):
        return self._mps

    @property
    def tuning(self):
        return self._tuning

    @property
    def trials(self):
        return self._trials

    @property
    def mixed_precision(self):
        return self._mp

    @property
    def compile_model(self):
        return self._compile

    @property
    def channels_last(self):
        return self._channels_last

    @property
    def persistent_workers(self):
        return self._persistent_workers

    @property
    def optimized(self):
        return self._optimized

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def separable(self):
        return self._separable

    @property
    def scheduler_enabled(self):
        return self._scheduler_enabled

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def pruning(self):
        return self._pruning

    @property
    def prune_amount(self):
        return self._prune_amount

    @property
    def preload(self):
        return self._preload

    @property
    def preload_path(self):
        return self._preload_path

    @property
    def seed(self):
        return self._seed

    @property
    def show_progress_bar(self):
        return self._pb

    @property
    def profiler(self):
        return self._profiler

    @property
    def show_result(self):
        return self._show_result

    @property
    def cycles(self):
        return self._cycles

    @property
    def device(self):
        return self._device

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_type = optimizer.__class__.__name__
        self.update_setting('optimizer', self._optimizer_type)

    @property
    def scaler(self):
        return self._scaler

    @property
    def training_data_loader(self):
        return self._training_data_loader

    @property
    def testing_data_loader(self):
        return self._testing_data_loader

    @property
    def validation_data_loader(self):
        return self._validation_data_loader

    @property
    def name(self):
        name = f"{self._upscale_factor}x_epochs({self._n_epochs})"
        if self._optimized:
            name += f"_optimized({self._num_classes})"
        if self._cuda:
            name += "_cuda"
        if self._mps:
            name += "_mps"
        if self._tuning:
            name += "_tuning"
        if self._pruning:
            name += f"_pruning({self._prune_amount})"
        if self._mp:
            name += "_mixed_precision"
        if self._scheduler_enabled:
            name += "_with_scheduler"
        if self._separable:
            name += "_separable"
        if self._optimizer_type:
            name += f"_optimizer({self._optimizer_type})"
        if self._preload:
            name += "_preloaded"
        name += f"_seed({self._seed})_batch_size({self._batch_size})"
        return name

    def create_model(self):
        if self.optimized:
            from model_ench import ObjectAwareESPCN as espcn
        else:
            from model import ESPCN as espcn
        model = espcn(upscale_factor=self.upscale_factor, num_classes=self.num_classes, separable=self.separable)
        
        # Apply channels_last memory format if enabled
        if self._channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        model = model.to(self.device)
        
        if self.preload and exists(self.preload_path):
            model = torch.load(self.preload_path, weights_only=False, map_location=self.device)
        
        return model

    @property
    def model_dir(self):
        return f'{join(self.model_path, self.name)}/'
    
    @classmethod
    def reset(cls):
        """Reset the Settings singleton to allow fresh initialization.
        
        Useful for:
            - Testing: Reset state between test cases
            - Hyperparameter tuning: Create fresh config for each trial
            - Hot-reloading: Reload settings.yaml without restart
        
        Usage:
            Settings.reset()
            new_settings = Settings()  # Fresh instance with reloaded config
        """
        Singleton.reset(cls)


def instance():
    return Settings()


def reset_instance():
    """Reset and return a fresh Settings instance.
    
    Convenience function for testing and tuning loops.
    """
    Settings.reset()
    return Settings()


def model_dir_i():
    return instance().model_dir

