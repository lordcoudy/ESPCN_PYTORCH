import os
from contextlib import nullcontext
from math import log10
from os import remove
from typing import Optional, Tuple

import progress.bar
import torch
import torch.profiler
from colorama import Fore
from torch.profiler import ProfilerActivity

from custom_logger import get_logger
from utils import calculate_ssim, measure_time

logger = get_logger('training')

@measure_time
def test(settings, bar, epoch) -> Tuple[float, float]:
    """Evaluate model on test set.
    
    Returns:
        Tuple of (average PSNR, average SSIM)
    """
    settings.model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = len(settings.testing_data_loader)
    
    # Use inference_mode for faster inference (more aggressive optimizations than no_grad)
    with torch.inference_mode():
        for test_iteration, batch in enumerate(settings.testing_data_loader, 1):
            if settings.show_progress_bar:
                bar.bar_prefix = f'Testing epoch {epoch + 1} [{test_iteration}/{num_batches}]: '
                bar.next()
            
            input_tensor = batch[0].to(settings.device, non_blocking=True)
            target_tensor = batch[1].to(settings.device, non_blocking=True)
            
            # Apply channels_last format if enabled
            if settings.channels_last:
                input_tensor = input_tensor.to(memory_format=torch.channels_last)
                target_tensor = target_tensor.to(memory_format=torch.channels_last)
            
            # Forward pass
            output = settings.model(input_tensor)
            
            # Calculate metrics
            mse = torch.nn.functional.mse_loss(output, target_tensor)
            psnr = 10 * log10(1.0 / (mse.item() + 1e-10))
            ssim = calculate_ssim(output, target_tensor)
            
            total_psnr += psnr
            total_ssim += ssim
            
            logger.debug(f"PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f}")

    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    logger.info(f"Avg. PSNR: {avg_psnr:.4f} dB | Avg. SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

@measure_time
def train_model(settings):
    """Main training loop with validation and checkpointing."""
    import copy

    from utils import (backPropagate, calculateLoss, checkpoint, export_model,
                       get_params, prune_model)
    
    settings.model.train()
    best_psnr = -float('inf')  # Start with -inf so first epoch always improves
    best_ssim = 0.0
    best_val_loss = float('inf')
    best_model_state = None  # Store best model weights
    patience_counter = 0
    early_stop_counter = 0  # For overfitting detection
    has_reached_target = False  # Track if we've ever reached target PSNR
    prev_train_loss = float('inf')
    prev_val_loss = float('inf')
    
    # Progress bar setup
    total_iterations = settings.epochs_number * (
        len(settings.training_data_loader) + 
        len(settings.validation_data_loader) + 
        len(settings.testing_data_loader)
    )
    bar = None
    if settings.show_progress_bar:
        bar = progress.bar.IncrementalBar(
            max=total_iterations,
            suffix='[%(percent).3f%%] - [%(elapsed).2fs>%(eta).2fs - %(avg).2fs / it]'
        )
        bar.start()
    
    # Configure profiler (runs once at the start if enabled)
    profiler_context = nullcontext()
    if settings.profiler:
        activities = [ProfilerActivity.CPU]
        if settings.cuda and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        profiler_context = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_modules=True,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
        )
    
    with profiler_context as prof:
        # Create CUDA stream for async operations if available
        stream = None
        if settings.device.type == 'cuda':
            stream = torch.cuda.Stream()
        
        for epoch in range(settings.epochs_number):
            epoch_loss = 0.0
            epoch_val_loss = 0.0
            num_train_batches = len(settings.training_data_loader)
            
            # Training phase
            settings.model.train()
            for iteration, batch in enumerate(settings.training_data_loader, 1):
                # Use non_blocking for async data transfer
                data = batch[0].to(settings.device, non_blocking=True)
                target = batch[1].to(settings.device, non_blocking=True)
                
                if settings.channels_last:
                    data = data.to(memory_format=torch.channels_last)
                    target = target.to(memory_format=torch.channels_last)
                
                if settings.show_progress_bar:
                    bar.bar_prefix = f'Training epoch {epoch + 1} [{iteration}/{num_train_batches}]: '
                    bar.next()
                
                # Gradient accumulation: only zero grads on first step
                is_accumulating = hasattr(settings, '_gradient_accumulation_steps') and settings._gradient_accumulation_steps > 1
                should_step = not is_accumulating or (iteration % settings._gradient_accumulation_steps == 0)
                
                if not is_accumulating or iteration % settings._gradient_accumulation_steps == 1:
                    settings.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass and optimization
                if stream is not None:
                    with torch.cuda.stream(stream):
                        loss = calculateLoss(settings, data, target, settings.model)
                else:
                    loss = calculateLoss(settings, data, target, settings.model)
                
                epoch_loss += loss.item()
                backPropagate(settings, loss, settings.optimizer, should_step=should_step)
                
                # OneCycleLR must be stepped after each batch (but only when optimizer steps)
                if should_step and settings.scheduler_enabled:
                    settings.scheduler.step()
                
                # Profile only first few iterations
                if settings.profiler and prof is not None:
                    prof.step()
                
                logger.debug(f"Epoch[{epoch+1}]({iteration}/{num_train_batches}): Loss: {loss.item():.6f}")
            
            # Log profiler results once after first epoch
            if settings.profiler and prof is not None and epoch == 0:
                prof_logger = get_logger('profiler')
                prof_logger.info(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            
            # Validation phase
            settings.model.eval()
            num_val_batches = len(settings.validation_data_loader)
            with torch.inference_mode():
                for val_iteration, val_batch in enumerate(settings.validation_data_loader, 1):
                    val_data = val_batch[0].to(settings.device, non_blocking=True)
                    val_target = val_batch[1].to(settings.device, non_blocking=True)
                    
                    if settings.channels_last:
                        val_data = val_data.to(memory_format=torch.channels_last)
                        val_target = val_target.to(memory_format=torch.channels_last)
                    
                    val_output = settings.model(val_data)
                    val_loss = settings.criterion(val_output, val_target)
                    epoch_val_loss += val_loss.item()
                    
                    if settings.show_progress_bar:
                        bar.bar_prefix = f'Validating epoch {epoch + 1} [{val_iteration}/{num_val_batches}]: '
                        bar.next()
            
            avg_train_loss = epoch_loss / num_train_batches
            avg_val_loss = epoch_val_loss / num_val_batches
            logger.info(f"Epoch {epoch+1}/{settings.epochs_number} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # Test phase
            test_psnr, test_ssim = test(settings, bar, epoch)
            
            # Track if we've reached target PSNR
            if test_psnr >= settings.target_min_psnr:
                has_reached_target = True
            
            # Early stopping check (PSNR-based)
            improved = test_psnr > best_psnr + settings.psnr_delta
            if improved:
                best_psnr = test_psnr
                best_ssim = test_ssim
                patience_counter = 0
                # Save best model state
                best_model_state = copy.deepcopy(settings.model.state_dict())
            else:
                patience_counter += 1
            
            # Track best validation loss for overfitting detection
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # Overfitting detection: val loss increasing while train loss decreasing
            is_overfitting = (
                avg_train_loss < prev_train_loss * 0.99 and  # Train loss still decreasing
                avg_val_loss > prev_val_loss * 1.01  # Val loss increasing
            )
            if is_overfitting:
                logger.warning(f"Potential overfitting detected: train loss ↓ ({prev_train_loss:.6f} → {avg_train_loss:.6f}), val loss ↑ ({prev_val_loss:.6f} → {avg_val_loss:.6f})")
            
            prev_train_loss = avg_train_loss
            prev_val_loss = avg_val_loss
            
            # Early stopping: exit if no improvement for patience epochs (after reaching target)
            if settings.early_stopping and has_reached_target:
                if early_stop_counter >= settings.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs (no val improvement for {settings.early_stopping_patience} epochs)")
                    logger.info(f"Best PSNR: {best_psnr:.4f} dB | Best SSIM: {best_ssim:.4f}")
                    # Restore best model weights
                    if best_model_state is not None:
                        settings.model.load_state_dict(best_model_state)
                        logger.info("Restored best model weights")
                        checkpoint(settings, settings.model, epoch + 1)
                        export_model(settings, settings.model, epoch + 1)
                    break
            
            # Only restart if:
            # 1. We've been stuck for stuck_level epochs AND
            # 2. We've NEVER reached target_min_psnr (not just current epoch)
            if patience_counter >= settings.stuck_level and not has_reached_target:
                logger.error(f"Training stuck for {settings.stuck_level} epochs below target {settings.target_min_psnr} dB. Restarting.")
                max_psnrs_path = os.path.join(settings.model_dir, 'max_psnrs.txt')
                if os.path.exists(max_psnrs_path):
                    remove(max_psnrs_path)
                return -2
            
            # Log patience status if stuck but above target (without early stopping)
            if patience_counter >= settings.stuck_level and has_reached_target and not settings.early_stopping:
                logger.info(f"No improvement for {patience_counter} epochs, but PSNR ({best_psnr:.2f} dB) already above target ({settings.target_min_psnr} dB). Continuing...")
            
            # Pruning
            if settings.pruning and (epoch + 1) % 200 == 0:
                prune_model(settings.model, settings.prune_amount)
            
            # Checkpointing
            if improved or (epoch + 1) == settings.checkpoint_frequency:
                checkpoint(settings, settings.model, epoch + 1)
                export_model(settings, settings.model, epoch + 1)
                
                os.makedirs(settings.model_dir, exist_ok=True)
                with open(os.path.join(settings.model_dir, 'max_psnrs.txt'), 'a+') as f:
                    print(f"Epoch {epoch+1}: PSNR={test_psnr:.4f} dB, SSIM={test_ssim:.4f}", file=f)
                
                if settings.show_progress_bar:
                    bar.suffix = f'[%(percent).3f%%] - Best PSNR: {best_psnr:.4f} dB'
    
    if settings.show_progress_bar and bar:
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
