from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ESPCN
# from model_ench import EnhancedESPCN as ESPCN
from data import get_training_set, get_test_set

from PIL import Image
from torchvision.transforms import ToTensor

import optuna

cuda = True
threads = 10
seed = 123

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without cuda")

torch.manual_seed(seed)

if cuda:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")

def train(epoch, training_data_loader, optimizer, criterion, model):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)


def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    upscale_factor=2
    model = ESPCN(upscale_factor=upscale_factor).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set = get_training_set(upscale_factor)
    dataloader = DataLoader(dataset=train_set, batch_size=batch_size)

    final_validation_loss = train(50, dataloader, optimizer, criterion, model)

    return final_validation_loss  # return the validation loss

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # perform the hyperparameter tuning

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

if __name__ == '__main__':
    main()
