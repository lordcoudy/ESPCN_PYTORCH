import time
from math import log10

import numpy as np
import progress.bar
import torch
from PIL import Image
from torch.nn import functional as f
from torch.nn.utils import prune
from torch.nn.utils.prune import L1Unstructured
from torchvision import models
from torchvision.models import VGG16_Weights
from torchvision.transforms import ToTensor


def checkpoint(settings, epoch):
    model_path = (settings.model_path + f"{settings.upscale_factor}x_epoch_{epoch}_{settings.model_name}.pth")
    torch.save(settings.model, model_path)
    print("===> Checkpoint saved to {} >===".format(model_path))


def perceptual_loss(device, pred, target):
    # Perceptual Loss with VGG16
    vgg = models.vgg16(weights = VGG16_Weights.DEFAULT).features[:16].eval().to(device)
    for param in vgg.parameters():
        param.requires_grad = False
    pred_vgg = vgg(pred.repeat(1, 3, 1, 1))
    target_vgg = vgg(target.repeat(1, 3, 1, 1))
    return f.mse_loss(pred_vgg, target_vgg)


def export_model(settings, epoch):
    settings.model.eval()
    img = Image.open(settings.input_path.convert('YCbCr'))
    y, cb, cr = img.split()
    img_to_tensor = ToTensor()
    input_tensor = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
    if settings.cuda and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    traced_script = torch.jit.trace(settings.model, input_tensor)
    traced_model_path = "{}x_traced_espcn_epoch_{}.pt".format(settings.upscale_factor, epoch)
    traced_script.save(traced_model_path)
    print("===> Model exported >===")
    print("===> Traced model saved to {}".format(traced_model_path))


def prune_model(model, amount = 0.2):
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.conv3, 'weight'),
        (model.conv4, 'weight')
    )
    prune.global_unstructured(
            parameters = parameters_to_prune,
            pruning_method = L1Unstructured,
            amount = amount,
    )


def test(settings):
    settings.model.eval()
    avg_psnr = 0
    max_mse = 0
    min_mse = 1
    with torch.no_grad():
        for batch in settings.testing_data_loader:
            input_tensor, target_tensor = batch[0].to(settings.device), batch[1].to(settings.device)
            output = settings.model(input_tensor)
            mse = settings.criterion(output, target_tensor)
            max_mse = max(max_mse, mse.item())
            min_mse = min(min_mse, mse.item())
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print(f"===> Avg. PSNR: {avg_psnr / len(settings.testing_data_loader):.12f} dB >===")
    print(f"===> Max. MSE: {max_mse:.12f} >===")
    print(f"===> Min. MSE: {min_mse:.12f} >===")


def train_model(settings):
    settings.model.train()
    epoch_losses = np.zeros(settings.epochs_number + 1)
    for epoch in range(settings.epochs_number + 1):
        epoch_loss = 0
        bar = progress.bar.IncrementalBar(f'Epoch {epoch + 1}', max = settings.training_data_loader.__len__())
        bar.start()
        for data, target in settings.training_data_loader:
            data, target = data.to(settings.device), target.to(settings.device)
            bar.next()
            settings.optimizer.zero_grad()

            # Mixed precision for GPU training
            with torch.amp.autocast('cuda', enabled = (settings.scaler is not None)):
                output = settings.model(data)
                mse = f.mse_loss(output, target)
                percLoss = perceptual_loss(settings.device, output, target)
                loss = mse + 0.1 * percLoss  # Combining losses
                # loss = criterion(output, target)

            if settings.scaler:
                settings.scaler.scale(loss).backward()
                settings.scaler.step(settings.optimizer)
                settings.scaler.update()
            else:
                loss.backward()
                settings.optimizer.step()

            epoch_loss += loss.item()
            epoch_losses[epoch] = epoch_loss

        bar.finish()

        # Learning rate decay
        settings.scheduler.step()
        if epoch in [25, 50, 100, 200, 500, 1000, 2000]:
            test(settings)
            checkpoint(settings, epoch)
            export_model(settings, epoch)
        print(
                f"Epoch {epoch + 1}/{settings.epochs_number}, Loss: {epoch_loss / len(settings.training_data_loader):.6f}, LR: {settings.scheduler.get_last_lr()[0]:.12f}")


def train(settings):
    print(f"===> Upscale factor: {settings.upscale_factor} | Epochs: {settings.epochs_number} >===")
    print("===> Building model >===")
    print("Structure of the model: ", settings.model)
    # Training and timing
    start_time = time.time()
    train_model(settings)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time):.2f} seconds")
    # plt.plot(dictionary['epochs_number'], g_epochLosses)
    # plt.show()
