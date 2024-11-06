import time
from math import log10

import numpy as np
import progress
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn.utils import prune
from torch.nn.utils.prune import L1Unstructured
from torchvision import models
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToTensor
from torch.nn import functional as F

from settings import *

g_epochLosses = np.zeros(dictionary['epochs_number'] + 1)

def checkpoint(epoch, upscaleFactor):
    modelPath = (dictionary['model_path']+f"{dictionary['upscale_factor']}x_{dictionary['model']}.pth")
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

def exportModel(epoch, upscaleFactor):
    model.eval()
    img = Image.open(dictionary['input_path']).convert('YCbCr')
    y, cb, cr = img.split()
    imgToTensor = ToTensor()
    input_tensor = imgToTensor(y).view(1, -1, y.size[1], y.size[0])
    if dictionary['cuda'] and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    tracedScript = torch.jit.trace(model, input_tensor)
    tracedModelPath = "{}x_traced_espcn_epoch_{}.pt".format(upscaleFactor, epoch)
    tracedScript.save(tracedModelPath)
    print("===> Model exported >===")
    print("===> Traced model saved to {}".format(tracedModelPath))

def pruneModel(amount=0.2):
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.conv3, 'weight'),
        (model.conv4, 'weight')
    )
    prune.global_unstructured(
        parameters=parameters_to_prune,
        pruning_method=L1Unstructured,
        amount=amount,
    )

def test(testingDataLoader):
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


def trainModel(dataloader, testloader, epochs, upscaleFactor=2):
    model.train()
    for epoch in range(epochs+1):
        epochLoss = 0
        bar = progress.bar.IncrementalBar(f'Epoch {epoch+1}', max=dictionary['batch_size'])
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
            test(testloader)
            checkpoint(epoch, upscaleFactor)
            exportModel(epoch, upscaleFactor)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss / len(dataloader):.6f}, LR: {scheduler.get_last_lr()[0]:.12f}")

def train():
    print(f"===> Upscale factor: {dictionary['upscale_factor']} | Epochs: {dictionary['epochs_number']} >===")
    # Data loaders
    # Model and optimizer
    print("===> Building model >===")
    print("Structure of the model: ", model)
    # optimizer = optim.Adam(model.parameters(), lr=g_lr)
    # Training and timing
    start_time = time.time()
    trainModel(trainLoader, testLoader, dictionary['epochs_number'], dictionary['upscale_factor'])
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time):.2f} seconds")
    plt.plot(dictionary['epochs_number'], g_epochLosses, 'o')
    plt.show()

