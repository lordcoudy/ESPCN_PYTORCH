from __future__ import print_function

import os

import optuna
import torch.optim as optim
from torch.utils.data import DataLoader

from data import get_training_set
from utils import backPropagate, calculateLoss, measure_time
from custom_logger import get_logger

logger = get_logger('tuning')

def train_model(settings, training_data_loader, model, optimizer):
    epoch_loss = 0
    logger.info(f"Train trial with lr: {settings.learning_rate}, batch_size: {settings.batch_size}, momentum: {settings.momentum}, weight_decay: {settings.weight_decay}, optimizer: {settings.optimizer}")
    for iteration, batch in enumerate(training_data_loader, 1):
        data, target = batch[0].to(settings.device), batch[1].to(settings.device)
        optimizer.zero_grad()
        loss = calculateLoss(settings, data, target, model)
        epoch_loss += loss.item()
        backPropagate(settings, loss, optimizer)
        logger.debug(f"===> {iteration}/{len(settings.training_data_loader)}): Loss: {loss.item():.6f}")
    logger.info(f"===> Avg. Loss: {epoch_loss / len(training_data_loader):.6f}")
    return epoch_loss / len(training_data_loader)

@measure_time
def objective(trial, settings):
    lr = trial.suggest_float('lr', 1e-7, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    momentum = trial.suggest_float('momentum', 0.9, 0.99)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3)
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])

    train_set = get_training_set(upscale_factor=settings.upscale_factor)
    dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True if settings.device == 'cuda' else False)

    model = settings.create_model()
    model.to(settings.device)

    optimizer_tuning = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == 'SGD':
        optimizer_tuning = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    final_validation_loss = train_model(settings, dataloader, model, optimizer_tuning)
    return final_validation_loss

@measure_time
def tune(settings):
    study = optuna.create_study(direction="minimize", study_name="ESPCN tuning")
    study.optimize(lambda trial: objective(trial, settings), n_trials=settings.trials, show_progress_bar=settings.show_progress_bar)

    logger.info('Best trial:')
    trial = study.best_trial
    logger.debug('\tValue: ', trial.value)
    logger.debug('\tParams: ')
    for key, value in trial.params.items():
        logger.debug(f'\t{key}: {value}')
        if key == 'lr':
            settings.learning_rate = value
        if key == 'batch_size':
            settings.batch_size = value
        if key == 'momentum':
            settings.momentum = value
        if key == 'weight_decay':
            settings.weight_decay = value
        if key == 'optimizer':
            optimizer = value
            settings.optimizer = optim.Adam(settings.model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
            if optimizer == 'SGD':
                settings.optimizer = optim.SGD(settings.model.parameters(), lr=settings.learning_rate, momentum=settings.momentum, weight_decay=settings.weight_decay)