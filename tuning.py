from __future__ import print_function

import optuna
import torch
import torch.nn.functional as f
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights

from data import get_training_set


def perceptual_loss(device, pred, target):
    # Perceptual Loss with VGG16
    vgg = models.vgg16(weights = VGG16_Weights.DEFAULT).features[:16].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    pred_vgg = vgg(pred.repeat(1, 3, 1, 1))
    target_vgg = vgg(target.repeat(1, 3, 1, 1))
    return f.mse_loss(pred_vgg, target_vgg)


def train_model(settings, training_data_loader, optimizer):
    settings.model.train()
    epoch_loss = 0
    for iteration, (input, target) in enumerate(training_data_loader, 1):
        input, target = input.to(settings.device), target.to(settings.device)
        optimizer.zero_grad()

        # Mixed precision for GPU training
        with torch.amp.autocast('cuda', enabled = (settings.scaler is not None)):
            output = settings.model(input)
            mse = f.mse_loss(output, target)
            perc_loss = perceptual_loss(settings.device, output, target)
            loss = mse + 0.1 * perc_loss  # Combining losses

        if settings.scaler:
            settings.scaler.scale(loss).backward()
            settings.scaler.step(optimizer)
            settings.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(training_data_loader)


def objective(settings, trial):
    lr = trial.suggest_float('lr', 1e-9, 1e-4, log = True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    train_set = get_training_set(upscale_factor = settings.upscale_factor)
    dataloader = DataLoader(dataset = train_set, batch_size = batch_size)
    optimizer_tuning = optim.SGD(settings.model.parameters(), lr = lr, momentum = 0.9)
    final_validation_loss = train_model(settings, dataloader, optimizer_tuning)
    return final_validation_loss  # return the validation loss


def tune(settings):
    study = optuna.create_study()
    study.optimize(objective(settings), n_trials = settings.trials,
                   show_progress_bar = settings.pb)  # perform the hyperparameter tuning
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
        settings.lr = key
