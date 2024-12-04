import progress.bar
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from data import get_test_set, get_training_set


class Singleton(type):
    _instances = { }

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# Create a singleton instance of the settings
def instance():
    settings = Settings()
    return settings


class Settings(metaclass = Singleton):
    def __init__(self):
        bar = progress.bar.IncrementalBar('Initializing ', max = 32)
        bar.start()

        stream = open("settings.yaml", 'r')
        bar.next()
        self.dictionary = yaml.load(stream, Loader = Loader)
        bar.next()

        self._upscale_factor = self.dictionary['upscale_factor']
        bar.next()
        self._n_epochs = self.dictionary['epochs_number']
        bar.next()
        self._model_path = self.dictionary['model_path']
        bar.next()
        self._input_path = self.dictionary['input_path']
        bar.next()
        self._output_path = self.dictionary['output_path']
        bar.next()
        self._threads = self.dictionary['threads']
        bar.next()
        self._batch_size = self.dictionary['batch_size']
        bar.next()
        self._test_batch_size = self.dictionary['test_batch_size']
        bar.next()
        self._seed = self.dictionary['seed']
        bar.next()
        self._lr = self.dictionary['learning_rate']
        bar.next()
        self._mp = self.dictionary['mixed_precision']
        bar.next()
        self._mode = self.dictionary['mode']
        bar.next()
        self._trials = self.dictionary['trials']
        bar.next()
        self._pb = self.dictionary['show_progress_bar']
        bar.next()
        self._epoch = self.dictionary['epoch']
        bar.next()
        self._pruning = self.dictionary['pruning']
        bar.next()
        self._tuning = self.dictionary['tuning']
        bar.next()

        train_set = get_training_set(self._upscale_factor)
        bar.next()
        test_set = get_test_set(self._upscale_factor)
        bar.next()
        if self.dictionary['optimized']:
            from model_ench import OptimizedESPCN as espcn
            self._pruning = False
        else:
            from model import ESPCN as espcn
        bar.next()
        self._cuda = self.dictionary['cuda']
        bar.next()
        self._device = torch.device("cuda" if self._cuda and torch.cuda.is_available() else "cpu")
        bar.next()
        torch.manual_seed(self._seed)
        bar.next()
        self._model = espcn().to(self._device)
        if self._cuda and torch.cuda.is_available():
            self._model = self._model.cuda()
        bar.next()
        self._criterion = nn.MSELoss()
        if self._cuda and torch.cuda.is_available():
            self._criterion = self._criterion.cuda()
        bar.next()
        self._optimizer = optim.SGD(self._model.parameters(), lr = self._lr, momentum = 0.9)
        bar.next()
        self._scheduler_enabled = self.dictionary['scheduler']
        if self._scheduler_enabled:
            self._scheduler = optim.lr_scheduler.StepLR(self._optimizer, step_size = 5, gamma = 0.5)
        bar.next()
        self._scaler = torch.amp.GradScaler(enabled = self._mp)
        bar.next()

        self._training_data_loader = DataLoader(dataset = train_set, num_workers = self._threads,
                                                batch_size = self._batch_size, shuffle = True)
        bar.next()
        self._testing_data_loader = DataLoader(dataset = test_set, num_workers = self._threads,
                                               batch_size = self._test_batch_size, shuffle = False)
        bar.next()
        bar.finish()

    @property
    def upscale_factor(self):
        return self._upscale_factor

    @property
    def epochs_number(self):
        return self._n_epochs

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
    def learning_rate(self):
        return self._lr

    @learning_rate.setter
    def learning_rate(self, lr : float):
        self._lr = lr

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
    def pruning(self):
        return self._pruning

    @property
    def tuning(self):
        return self._tuning

