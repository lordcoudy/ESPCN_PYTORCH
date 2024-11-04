from __future__ import print_function

import time
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import progress.bar

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from data import getTrainingSet, getTestSet
# from model import ESPCN as espcn
from model_ench import AltESPCN as espcn
# from model_ench import OptimizedESPCN as espcn

# Hyperparameters
g_inputPath = 'E:\\SAVVA\\STUDY\\CUDA\\ESPCN_PYTORCH\\dataset\\BSDS300\\images\\test\\3096.jpg'
# g_upscaleFactors = [ 2, 3, 4, 8 ]
g_upscaleFactors = [2]
g_batchSize = 16
g_testBatchSize = 16
g_nEpochs = 1000
g_epochLosses = np.zeros(g_nEpochs+1)
g_lr = 0.0268712201875209
g_threads = 8
g_seed = 123
g_cuda = True

# Device configuration
device = torch.device("cuda" if g_cuda and torch.cuda.is_available() else "cpu")
torch.manual_seed(g_seed)

g_criterion = nn.MSELoss()
# Training with mixed precision and scheduler
g_scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

def prepareData(upscaleFactor):
    print('===> Loading datasets >===')
    trainSet = getTrainingSet(upscaleFactor)
    testSet = getTestSet(upscaleFactor)
    trainingDataLoader = DataLoader(dataset=trainSet, num_workers=g_threads, batch_size=g_batchSize, shuffle=True)
    testingDataLoader = DataLoader(dataset=testSet, num_workers=g_threads, batch_size=g_testBatchSize, shuffle=False)
    print('===> Datasets loaded >===')
    return trainingDataLoader, testingDataLoader


def checkpoint(model, epoch, upscaleFactor):
    modelPath = "{}x_espcn_epoch_{}.pth".format(upscaleFactor, epoch)
    torch.save(model, modelPath)
    print("===> Checkpoint saved to {} >===".format(modelPath))

def perceptualLoss(pred, target):
    # Perceptual Loss with VGG16
    vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    pred_vgg = vgg(pred.repeat(1, 3, 1, 1))
    target_vgg = vgg(target.repeat(1, 3, 1, 1))
    return nn.functional.mse_loss(pred_vgg, target_vgg)

def exportModel(model, epoch, upscaleFactor, inputImage):
    model.eval()
    img = Image.open(inputImage).convert('YCbCr')
    y, cb, cr = img.split()
    imgToTensor = ToTensor()
    if g_cuda and torch.cuda.is_available():
        input_tensor = imgToTensor(y).view(1, -1, y.size[1], y.size[0]).cuda()
    else:
        input_tensor = imgToTensor(y).view(1, -1, y.size[1], y.size[0])

    tracedScript = torch.jit.trace(model, input_tensor)
    tracedModelPath = "{}x_traced_espcn_epoch_{}.pt".format(upscaleFactor, epoch)
    tracedScript.save(tracedModelPath)
    print("===> Model exported >===")
    print("===> Traced model saved to {}".format(tracedModelPath))

def pruneModel(model, amount=0.2):
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.conv3, 'weight'),
        (model.conv4, 'weight')
    )
    prune.global_unstructured(
        parameters=parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

def test(model, testingDataLoader):
    model.eval()
    avgPSNR = 0
    max_mse = 0
    min_mse = 1
    with torch.no_grad():
        for batch in testingDataLoader:
            input_tensor, target_tensor = batch[0].to(device), batch[1].to(device)
            output = model(input_tensor)
            mse = g_criterion(output, target_tensor)
            max_mse = max(max_mse, mse.item())
            min_mse = min(min_mse, mse.item())
            psnr = 10 * log10(1 / mse.item())
            avgPSNR += psnr
    print(f"===> Avg. PSNR: {avgPSNR / len(testingDataLoader):.12f} dB >===")
    print(f"===> Max. MSE: {max_mse:.12f} >===")
    print(f"===> Min. MSE: {min_mse:.12f} >===")


def trainModel(model, dataloader, testloader, criterion, optimizer, scheduler, epochs, scaler=None, upscaleFactor=2):
    model.train()
    for epoch in range(epochs+1):
        epochLoss = 0
        bar = progress.bar.IncrementalBar(f'Epoch {epoch+1}', max=g_batchSize)
        bar.start()
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            bar.next()
            optimizer.zero_grad()

            # Mixed precision for GPU training
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                output = model(data)
                mse = F.mse_loss(output, target)
                percLoss = perceptualLoss(output, target)
                loss = mse + 0.1 * percLoss  # Combining losses
                # loss = criterion(output, target)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epochLoss += loss.item()
            g_epochLosses[epoch]=epochLoss

        bar.finish()

        # Learning rate decay
        scheduler.step()
        if epoch in [25, 50, 100, 200, 500, 1000, 2000]:
            test(model, testloader)
            checkpoint(model, epoch, upscaleFactor)
            exportModel(model, epoch, upscaleFactor, g_inputPath)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss / len(dataloader):.6f}, LR: {scheduler.get_last_lr()[0]:.12f}")

def main():
    print("Cuda is available: ", torch.cuda.is_available())
    for upscaleFactor in g_upscaleFactors:
        print(f"===> Upscale factor: {upscaleFactor} | Epochs: {g_nEpochs} >===")
        # Data loaders
        trainLoader, testLoader = prepareData(upscaleFactor)
        # Model and optimizer
        print("===> Building model >===")
        model = espcn(upscaleFactor=upscaleFactor, numChannels=1).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=g_lr)
        optimizer = optim.SGD(model.parameters(), lr=g_lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Training and timing
        start_time = time.time()
        trainModel(model, trainLoader, testLoader, g_criterion, optimizer, scheduler, g_nEpochs, g_scaler, upscaleFactor)
        end_time = time.time()
        print(f"Training completed in {(end_time - start_time):.2f} seconds")
        plt.plot(g_nEpochs, g_epochLosses, 'o')
        plt.show()


if __name__ == '__main__':
    main()
