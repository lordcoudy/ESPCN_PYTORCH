from __future__ import print_function
from math import log10
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# from model import ESPCN
from model_ench import OptimizedESPCN as ESPCN
# from model_ench import AltESPCN as ESPCN

from data import getTrainingSet, getTestSet

import optuna

# Hyperparameters
# g_upscaleFactors = [ 2, 3, 4, 8 ]
g_upscaleFactor = 2
g_nEpochs = 100
g_seed = 123
g_cuda = False

# Device configuration
device = torch.device("cuda" if g_cuda and torch.cuda.is_available() else "cpu")
torch.manual_seed(g_seed)

g_criterion = nn.MSELoss()
g_model = ESPCN(upscaleFactor=g_upscaleFactor).to(device)


def train(epoch, trainingDataLoader, optimizer):
    epochLoss = 0
    for iteration, batch in enumerate(trainingDataLoader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        output = g_model(input)
        loss = g_criterion(output, target)
        epochLoss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}/{}]({}/{}): Loss: {:.4f}".format(epoch, g_nEpochs, iteration, len(trainingDataLoader), loss.item()))

    print("===> Epochs {}/{} Complete: Avg. Loss: {:.4f}".format(epoch, g_nEpochs, epochLoss / len(trainingDataLoader)))
    return epochLoss / len(trainingDataLoader)


def objective(trial):
    lr = trial.suggest_float('lr', 1e-9, 1e-1, log=True)
    batchSize = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    optimizer = optim.Adam(g_model.parameters(), lr=lr)
    trainSet = getTrainingSet(g_upscaleFactor)
    dataloader = DataLoader(dataset=trainSet, batch_size=batchSize)

    finalValidationLoss = train(g_nEpochs, dataloader, optimizer)

    return finalValidationLoss  # return the validation loss

def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)  # perform the hyperparameter tuning

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

if __name__ == '__main__':
    main()
