from math import log10

import progress.bar
from colorama import Fore
import torch.profiler
from torch.profiler import ProfilerActivity
from contextlib import nullcontext

from utils import *
from custom_logger import get_logger

logger = get_logger('training')

@measure_time
def test(settings):
    settings.model.eval()
    avg_psnr = 0
    max_mse = 0
    min_mse = 1
    with torch.no_grad():
        for batch in settings.testing_data_loader:
            input_tensor, target_tensor = batch[0].to(settings.device), batch[1].to(settings.device)
            mse = calculateLoss(settings, input_tensor, target_tensor, settings.model)
            max_mse = max(max_mse, mse.item())
            min_mse = min(min_mse, mse.item())
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            logger.debug(f"PSNR: {psnr:.6f} dB | MSE: {mse.item():.6f}")
    avg_psnr /= len(settings.testing_data_loader)
    logger.info(f"===> Avg. PSNR: {avg_psnr:.12f} dB >===")
    logger.debug(f"===> Max. MSE: {max_mse:.12f} >===")
    logger.debug(f"===> Min. MSE: {min_mse:.12f} >===")
    return avg_psnr

@measure_time
def train_model(settings):
    settings.model.train()
    psnrs = [0]
    slowdown_counter = 0
    for epoch in range(settings.epochs_number):
        epoch_loss = 0
        epoch_val_loss = 0
        if settings.show_progress_bar:
            bar = progress.bar.IncrementalBar(f'Epoch {epoch + 1}', max=len(settings.training_data_loader) + len(settings.validation_data_loader)) # Update bar max
        bar.start()
        for iteration, batch in enumerate(settings.training_data_loader, 1):
            data, target = batch[0].to(settings.device), batch[1].to(settings.device)
            bar.next()
            prof = torch.profiler.profile(
                    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes = True,
                    profile_memory = True,
                    with_modules = True)
            with prof if settings.profiler else nullcontext():
                settings.optimizer.zero_grad()
                loss = calculateLoss(settings, data, target, settings.model)
                epoch_loss += loss.item()
                backPropagate(settings, loss, settings.optimizer)
            if settings.profiler: logger.debug(prof.key_averages().table(sort_by = "self_cuda_memory_usage"))
            logger.debug(f"===> Epoch[{epoch+1}]({iteration}/{len(settings.training_data_loader)}): Loss: {loss.item():.6f}")

        settings.model.eval()
        with torch.no_grad():
            for val_iteration, val_batch in enumerate(settings.validation_data_loader, 1):
                val_data, val_target = val_batch[0].to(settings.device), val_batch[1].to(settings.device)
                val_loss = calculateLoss(settings, val_data, val_target, settings.model)
                epoch_val_loss += val_loss.item()
                bar.next()
        settings.model.train()
        if settings.scheduler_enabled:
            settings.scheduler.step(epoch_val_loss if settings.scheduler == optim.lr_scheduler.ReduceLROnPlateau else None)
        logger.info(f"===> Epoch {epoch+1}/{settings.epochs_number} Complete: Avg. Loss: {epoch_loss / len(settings.training_data_loader):.12f} Avg. Val. Loss: {epoch_val_loss / len(settings.validation_data_loader)}")
        bar.finish()
        t_psnr = test(settings)
        psnrs.append(t_psnr)
        delta = psnrs[-1] - max(psnrs)
        if delta < settings.psnr_delta:
            slowdown_counter += 1
        else:
            slowdown_counter = 0
        if slowdown_counter == settings.stuck_level and t_psnr < settings.target_min_psnr:
            logger.error(Fore.RED + f"===> Training seems to be stuck. Rerunning. >===")
            return -2

        if settings.pruning and (epoch + 1) % 100 == 0:
            prune_model(settings.model, settings.prune_amount)

        if t_psnr == max(psnrs) or epoch+1 == settings.checkpoint_frequency:
            checkpoint(settings, settings.model, epoch+1)
            export_model(settings, settings.model, epoch+1)
            os.makedirs('psnrs', exist_ok=True)
            with open(os.path.join('psnrs', 'max_psnrs.txt'), 'a+') as f:
                print(f"Epoch {epoch+1}: {t_psnr:.6f} dB", end="\n", file=f)
    get_params(settings.model)


def train(settings):
    logger.info(f"===> Upscale factor: {settings.upscale_factor} | Epochs: {settings.epochs_number} >===")
    logger.debug(f"===> Batch size: {settings.batch_size} | Learning rate: {settings.learning_rate} | Weight decay: {settings.weight_decay} | Optimizer: {settings.optimizer_type} >===")
    logger.info("===> Building model >===")
    # logger.debug("Structure of the model: ", settings.model)
    os.makedirs('times', exist_ok=True)
    with open(os.path.join('times', 'time_train_model.txt'), 'a+') as f:
        print(f"{settings.name}: ", end="\n", file=f)
    return train_model(settings)
