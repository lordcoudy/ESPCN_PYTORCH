from os.path import exists

import numpy
import progress.bar
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from data import get_test_set, get_training_set, get_validation_set


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Settings(metaclass=Singleton):
    def __init__(self):
        stream = open("settings.yaml", 'r')
        self.dictionary = yaml.load(stream, Loader=Loader)
        self._pb = self.dictionary['show_progress_bar']
        if self._pb:
            bar = progress.bar.IncrementalBar('Initializing ', max=49, suffix='%(percent)d%%')
            bar.start()
            bar.next()
            bar.next()

        self._upscale_factor = self.dictionary['upscale_factor']
        if self._pb:
            bar.next()
        self._n_epochs = self.dictionary['epochs_number']
        if self._pb:
            bar.next()
        self._checkpoint_freq = self.dictionary['checkpoint_frequency']
        if self._pb:
            bar.next()
        self._model_path = self.dictionary['model_path']
        if self._pb:
            bar.next()
        self._input_path = self.dictionary['input_path']
        if self._pb:
            bar.next()
        self._output_path = self.dictionary['output_path']
        if self._pb:
            bar.next()
        self._threads = self.dictionary['threads']
        if self._pb:
            bar.next()
        self._batch_size = self.dictionary['batch_size']
        if self._pb:
            bar.next()
        self._test_batch_size = self.dictionary['test_batch_size']
        if self._pb:
            bar.next()
        self._seed = self.dictionary['seed']
        if self._pb:
            bar.next()
        self._num_classes = self.dictionary['num_classes']
        if self._pb:
            bar.next()
        self._lr = self.dictionary['learning_rate']
        if self._pb:
            bar.next()
        self._momentum = self.dictionary['momentum']
        if self._pb:
            bar.next()
        self._weight_decay = self.dictionary['weight_decay']
        if self._pb:
            bar.next()
        self._mp = self.dictionary['mixed_precision']
        if self._pb:
            bar.next()
        self._mode = self.dictionary['mode']
        if self._pb:
            bar.next()
        self._trials = self.dictionary['trials']
        if self._pb:
            bar.next()
        self._epoch = self.dictionary['epoch']
        if self._pb:
            bar.next()
        self._pruning = self.dictionary['pruning']
        if self._pb:
            bar.next()
        self._tuning = self.dictionary['tuning']
        if self._pb:
            bar.next()
        self._optimizer_type = self.dictionary['optimizer']
        if self._pb:
            bar.next()
        self._optimized = self.dictionary['optimized']
        if self._pb:
            bar.next()
        self._separable = self.dictionary['separable']
        if self._pb:
            bar.next()
        self._preload = self.dictionary['preload']
        if self._pb:
            bar.next()
        self._preload_path = self.dictionary['preload_path']
        if self._pb:
            bar.next()
        self._psnr_delta = self.dictionary['psnr_delta']
        if self._pb:
            bar.next()
        self._stuck_level = self.dictionary['stuck_level']
        if self._pb:
            bar.next()
        self._target_min_psnr = self.dictionary['target_min_psnr']
        if self._pb:
            bar.next()
        train_set = get_training_set(self._upscale_factor)
        if self._pb:
            bar.next()
        test_set = get_test_set(self._upscale_factor)
        if self._pb:
            bar.next()
        val_set = get_validation_set(self._upscale_factor)
        if self._pb:
            bar.next()
        if self._optimized:
            from model_ench import ObjectAwareESPCN as espcn
            self._pruning = False
        else:
            from model import ESPCN as espcn
        if self._pb:
            bar.next()
        self._cuda = self.dictionary['cuda']
        if self._pb:
            bar.next()
        self._device = torch.device("cuda" if self._cuda and torch.cuda.is_available() else "cpu")
        if self._pb:
            bar.next()
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        torch.backends.cudnn.benchmark = True
        numpy.random.seed(self._seed)
        if self._pb:
            bar.next()
        self._model = espcn(self._num_classes, self._upscale_factor, separable = self._separable).to(self._device)
        if self._preload and exists(self._preload_path):
            self._model = torch.load(self._preload_path, weights_only=False, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if self._pb:
            bar.next()
        self._criterion = nn.MSELoss().to(self._device)
        if self._pb:
            bar.next()
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        if self._optimizer_type == 'SGD':
            self._optimizer = optim.SGD(self._model.parameters(), lr=self._lr, momentum=self._momentum, weight_decay=self._weight_decay)
        if self._pb:
            bar.next()
        device_type = 'cpu'
        if self._cuda:
            device_type = 'cuda'
        self._scaler = torch.amp.GradScaler(device=device_type, enabled=self._mp)
        if self._pb:
            bar.next()

        self._training_data_loader = DataLoader(dataset=train_set, num_workers=self._threads,
                                                batch_size=self._batch_size, shuffle=True)
        if self._pb:
            bar.next()
        self._testing_data_loader = DataLoader(dataset=test_set, num_workers=self._threads,
                                               batch_size=self._test_batch_size, shuffle=False)
        if self._pb:
            bar.next()
        self._validation_data_loader = DataLoader(dataset=val_set, num_workers=self._threads,
                                                  batch_size=self._test_batch_size, shuffle=True)
        if self._pb:
            bar.next()
        self._scheduler_enabled = self.dictionary['scheduler']
        if self._scheduler_enabled:
            self._scheduler = optim.lr_scheduler.OneCycleLR(
                self._optimizer,
                max_lr=self._lr,
                steps_per_epoch=len(self._training_data_loader),
                epochs=self._n_epochs,
                anneal_strategy='cos',
                final_div_factor=1e4
            )
        if self._pb:
            bar.next()
        self._prune_amount = self.dictionary['prune_amount']
        if self._pb:
            bar.next()
        self._profiler = self.dictionary['show_profiler']
        if self._pb:
            bar.next()
        self._show_result = self.dictionary['show_result']
        if self._pb:
            bar.next()
        self._cycles = self.dictionary['cycles']
        if self._pb:
            bar.next()
            bar.finish()

    @property
    def upscale_factor(self):
        return self._upscale_factor

    @property
    def epochs_number(self):
        return self._n_epochs

    @property
    def checkpoint_frequency(self):
        return self._checkpoint_freq

    @property
    def model_path(self):
        return self._model_path

    @property
    def input_path(self):
        return self._input_path

    @property
    def output_path(self):
        return self._output_path

    @property
    def threads(self):
        return self._threads

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def test_batch_size(self):
        return self._test_batch_size

    @property
    def seed(self):
        return self._seed

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, lr: float):
        self._lr = lr

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, momentum: float):
        self._momentum = momentum

    @property
    def weight_decay(self):
        return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, weight_decay: float):
        self._weight_decay = weight_decay

    @property
    def mixed_precision(self):
        return self._mp

    @property
    def mode(self):
        return self._mode

    @property
    def trials(self):
        return self._trials

    @property
    def show_progress_bar(self):
        return self._pb

    @property
    def epoch(self):
        return self._epoch

    @property
    def model(self):
        return self._model

    @property
    def cuda(self):
        return self._cuda

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

    @property
    def scheduler_enabled(self):
        return self._scheduler_enabled

    @property
    def scheduler(self):
        return self._scheduler

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
    def pruning(self):
        return self._pruning

    @property
    def tuning(self):
        return self._tuning

    @property
    def optimized(self):
        return self._optimized

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
    def psnr_delta(self):
        return self._psnr_delta

    @property
    def stuck_level(self):
        return self._stuck_level

    @property
    def target_min_psnr(self):
        return self._target_min_psnr

    @property
    def profiler(self):
        return self._profiler

    @property
    def optimizer_type(self):
        return self._optimizer_type

    @property
    def separable(self):
        return self._separable

    @property
    def name(self):
        name = f"{self._upscale_factor}x_epochs({self._n_epochs})"
        if self._optimized:
            name += f"_optimized({self._num_classes})"
        if self._cuda:
            name += "_cuda"
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

    @property
    def show_result(self):
        return self._show_result

    @property
    def cycles(self):
        return self._cycles

    def create_model(self):
        if self.optimized:
            from model_ench import ObjectAwareESPCN as espcn
        else:
            from model import ESPCN as espcn
        model = espcn(upscale_factor=self.upscale_factor, num_classes=self.num_classes, separable = self.separable).to(self.device)
        if self.preload and exists(self.preload_path):
            model = torch.load(self.preload_path, weights_only=False, map_location=self.device)
        return model


def instance():
    return Settings()