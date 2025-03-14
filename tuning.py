from __future__ import print_function

import os

import optuna
import torch.optim as optim
from data import get_training_set
from settings import Settings
from torch.utils.data import DataLoader
from utils import backPropagate, calculateLoss, measure_time


def train_model(settings, training_data_loader, optimizer):
    settings.model.train()
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        data, target = batch[0].to(settings.device), batch[1].to(settings.device)
        optimizer.zero_grad()
        loss = calculateLoss(settings, data, target)
        epoch_loss += loss.item()
        backPropagate(settings, loss)
    return epoch_loss / len(training_data_loader)

@measure_time
def objective(trial):
    lr = trial.suggest_float('lr', 1e-9, 1e-4, log = True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
    momentum = trial.suggest_float('momentum', 0.9, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3)
    settings = Settings()
    train_set = get_training_set(upscale_factor = settings.upscale_factor)
    dataloader = DataLoader(
        dataset = train_set,
        batch_size = batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True if settings.device == 'cuda' else False)
    model = settings.create_model()
    model.to(settings.device)
    # optimizer_tuning = optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay=weight_decay)
    optimizer_tuning = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    final_validation_loss = train_model(settings, dataloader, optimizer_tuning)
    return final_validation_loss

@measure_time
def tune(settings):
    study = optuna.create_study(direction="minimize", study_name="ESPCN tuning")
    study.optimize(objective, n_trials = settings.trials,
                   show_progress_bar = settings.show_progress_bar)
    print('Best trial:')
    trial = study.best_trial
    print('\tValue: ', trial.value)
    print('\tParams: ')
    for key, value in trial.params.items():
        print(f'\t{key}: {value}')
        if key == 'lr':
            settings.learning_rate = value
        if key == 'batch_size':
            settings.batch_size = value
        if key == 'momentum':
            settings.momentum = value
        if key == 'weight_decay':
            settings.weight_decay = value

    # settings.optimizer = optim.SGD(settings.model.parameters(), lr = settings.learning_rate, momentum= settings.momentum, weight_decay=settings.weight_decay)
    settings.optimizer = optim.Adam(settings.model.parameters(), lr = settings.learning_rate, weight_decay=settings.weight_decay)
