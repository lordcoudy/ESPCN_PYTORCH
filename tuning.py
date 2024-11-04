from __future__ import print_function

import optuna
import torch
import torch.optim as optim
from data import getTrainingSet
# from model import ESPCN as espcn
from model_ench import OptimizedESPCN as espcn
# from model_ench import AltESPCN as espcn
from run import g_scaler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


# Hyperparameters
# g_upscaleFactors = [ 2, 3, 4, 8 ]
g_upscaleFactor = 2
g_seed = 123
g_cuda = True

# Device configuration
device = torch.device("cuda" if g_cuda and torch.cuda.is_available() else "cpu")
torch.manual_seed(g_seed)


def perceptualLoss(pred, target):
    # Perceptual Loss with VGG16
    vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    pred_vgg = vgg(pred.repeat(1, 3, 1, 1))
    target_vgg = vgg(target.repeat(1, 3, 1, 1))
    return F.mse_loss(pred_vgg, target_vgg)

def trainModel(model, trainingDataLoader, optimizer, scaler=None):
    model.train()
    epochLoss = 0
    for iteration, (input, target) in enumerate(trainingDataLoader, 1):
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()

        # Mixed precision for GPU training
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            output = model(input)
            mse = F.mse_loss(output, target)
            percLoss = perceptualLoss(output, target)
            loss = mse + 0.1 * percLoss  # Combining losses

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epochLoss += loss.item()

    return epochLoss / len(trainingDataLoader)



def objective(trial):
    lr = trial.suggest_float('lr', 1e-9, 1e-1, log=True)
    batchSize = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    trainSet = getTrainingSet(g_upscaleFactor)
    dataloader = DataLoader(dataset=trainSet, batch_size=batchSize)
    model = espcn(upscaleFactor=g_upscaleFactor, numChannels=1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    finalValidationLoss = trainModel(model, dataloader, optimizer, g_scaler)

    return finalValidationLoss  # return the validation loss

def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100, show_progress_bar=True) # perform the hyperparameter tuning

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

if __name__ == '__main__':
    main()
