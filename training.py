import os
from contextlib import nullcontext
from math import log10
from os import remove

import progress.bar
import torch
import torch.profiler
from colorama import Fore
from torch.profiler import ProfilerActivity

from custom_logger import get_logger
from utils import *

logger = get_logger('training')

@measure_time
def test(settings, bar, epoch):
    settings.model.eval()
    avg_psnr = 0
    max_mse = 0
    min_mse = 1
    # Use inference_mode for faster inference (more aggressive optimizations than no_grad)
    with torch.inference_mode():
        for test_iteration, batch in enumerate(settings.testing_data_loader, 1):
            if settings.show_progress_bar:
                bar.bar_prefix = f'Testing epoch {epoch + 1} [{test_iteration}/{len(settings.testing_data_loader)}]: '
                bar.next()
            input_tensor, target_tensor = batch[0].to(settings.device), batch[1].to(settings.device)
            # Apply channels_last format if enabled
            if settings.channels_last:
                input_tensor = input_tensor.to(memory_format=torch.channels_last)
                target_tensor = target_tensor.to(memory_format=torch.channels_last)
            mse = calculateLoss(settings, input_tensor, target_tensor, settings.model)
            max_mse = max(max_mse, mse.item())
            min_mse = min(min_mse, mse.item())
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            logger.debug(f"PSNR: {psnr:.6f} dB | MSE: {mse.item():.6f}")

    avg_psnr /= len(settings.testing_data_loader)
    logger.info(f"Avg. PSNR: {avg_psnr:.12f} dB")
    logger.debug(f"Max. MSE: {max_mse:.12f}")
    logger.debug(f"Min. MSE: {min_mse:.12f}")
    return avg_psnr

@measure_time
def train_model(settings):
    settings.model.train()
    psnrs = [0]
    slowdown_counter = 0
    if settings.show_progress_bar:
        bar = progress.bar.IncrementalBar(max=settings.epochs_number*(len(settings.training_data_loader) + len(settings.validation_data_loader) + len(settings.testing_data_loader)),
                                                  suffix='[%(percent).3f%%] - [%(elapsed).2fs>%(eta).2fs - %(avg).2fs / it]') # Update bar max
        bar.start()
    for epoch in range(settings.epochs_number):
        epoch_loss = 0
        epoch_val_loss = 0
        for iteration, batch in enumerate(settings.training_data_loader, 1):
            # Use non_blocking for async data transfer (overlaps with computation)
            data = batch[0].to(settings.device, non_blocking=True)
            target = batch[1].to(settings.device, non_blocking=True)
            # Apply channels_last memory format if enabled
            if settings.channels_last:
                data = data.to(memory_format=torch.channels_last)
                target = target.to(memory_format=torch.channels_last)
            if settings.show_progress_bar:
                bar.bar_prefix = f'Training epoch {epoch + 1} [{iteration}/{len(settings.training_data_loader)}]: '
                bar.next()
            # Configure profiler activities based on device
            activities = [ProfilerActivity.CPU]
            if settings.cuda and torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            prof = torch.profiler.profile(
                    activities = activities,
                    record_shapes = True,
                    profile_memory = True,
                    with_modules = True)
            with prof if settings.profiler else nullcontext():
                settings.optimizer.zero_grad(set_to_none=True)  # More memory efficient
                loss = calculateLoss(settings, data, target, settings.model)
                epoch_loss += loss.item()
                backPropagate(settings, loss, settings.optimizer)
                # OneCycleLR must be stepped after each batch (not epoch)
                if settings.scheduler_enabled:
                    settings.scheduler.step()
            if settings.profiler:
                prof_logger = get_logger('profiler')
                prof_logger.info(prof.key_averages().table(sort_by = "self_cuda_memory_usage", row_limit = 10))
            logger.debug(f"Epoch[{epoch+1}]({iteration}/{len(settings.training_data_loader)}): Loss: {loss.item():.6f}")
        settings.model.eval()
        # Use inference_mode for validation (faster than no_grad)
        with torch.inference_mode():
            for val_iteration, val_batch in enumerate(settings.validation_data_loader, 1):
                val_data = val_batch[0].to(settings.device, non_blocking=True)
                val_target = val_batch[1].to(settings.device, non_blocking=True)
                if settings.channels_last:
                    val_data = val_data.to(memory_format=torch.channels_last)
                    val_target = val_target.to(memory_format=torch.channels_last)
                val_loss = calculateLoss(settings, val_data, val_target, settings.model)
                epoch_val_loss += val_loss.item()
                if settings.show_progress_bar:
                    bar.bar_prefix = f'Validating epoch {epoch + 1} [{val_iteration}/{len(settings.validation_data_loader)}]: '
                    bar.next()
        settings.model.train()
        logger.info(f"Epoch {epoch+1}/{settings.epochs_number} Complete: Avg. Loss: {epoch_loss / len(settings.training_data_loader):.12f} Avg. Val. Loss: {epoch_val_loss / len(settings.validation_data_loader)}")
        t_psnr = test(settings, bar, epoch)
        psnrs.append(t_psnr)
        delta = t_psnr - max(psnrs)
        if delta < settings.psnr_delta:
            slowdown_counter += 1
        else:
            slowdown_counter = 0
        if slowdown_counter == settings.stuck_level and t_psnr < settings.target_min_psnr:
            logger.error(f"Training seems to be stuck. Rerunning.")
            remove(os.path.join(settings.model_dir, 'max_psnrs.txt'))
            return -2

        if settings.pruning and (epoch + 1) % 200 == 0:
            prune_model(settings.model, settings.prune_amount)

        if t_psnr == max(psnrs) or epoch+1 == settings.checkpoint_frequency:
            checkpoint(settings, settings.model, epoch+1)
            export_model(settings, settings.model, epoch+1)
            max_ = max(psnrs)
            os.makedirs(settings.model_dir, exist_ok = True)
            with open(os.path.join(settings.model_dir, 'max_psnrs.txt'), 'a+') as f:
                print(f"Epoch {epoch+1}: {t_psnr:.6f} dB", end="\n", file=f)
            if settings.show_progress_bar:
                bar.suffix = f'[%(percent).3f%%] - [%(elapsed).2fs>%(eta).2fs - %(avg).2fs / it] - Max.PSNR: {max_:.6f} dB'
    if settings.show_progress_bar:
        bar.finish()
    get_params(settings.model)
    return 0


def train(settings):
    logger.info(f"Upscale factor: {settings.upscale_factor} | Epochs: {settings.epochs_number}")
    logger.debug(f"Batch size: {settings.batch_size} | Learning rate: {settings.learning_rate} | Weight decay: {settings.weight_decay} | Optimizer: {settings.optimizer_type}")
    logger.info("Building model")
    logger.debug(f"Structure of the model: {settings.model}")
    os.makedirs('times', exist_ok=True)
    with open(os.path.join('times', 'time_train_model.txt'), 'a+') as f:
        print(f"{settings.name}: ", end="\n", file=f)
    return train_model(settings)
