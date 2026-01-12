"""
Automatic configuration module for ESPCN based on machine specifications.

This module detects hardware capabilities (CPU, RAM, GPU) and automatically
configures optimal training settings for best performance.
"""

import multiprocessing
import platform
import sys
from typing import Dict, Tuple

import psutil
import torch
import yaml

from custom_logger import get_logger

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

logger = get_logger('autoconfig')


class MachineSpecs:
    """Detects and stores machine hardware specifications."""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.physical_cores = psutil.cpu_count(logical=False) or self.cpu_count
        self.logical_cores = psutil.cpu_count(logical=True) or self.cpu_count
        self.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        self.platform = platform.system()
        self.platform_release = platform.release()
        self.python_version = sys.version_info
        
        # GPU Detection
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        if self.has_cuda:
            self.cuda_device_count = torch.cuda.device_count()
            self.cuda_device_name = torch.cuda.get_device_name(0) if self.cuda_device_count > 0 else None
            self.cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if self.cuda_device_count > 0 else 0
        else:
            self.cuda_device_count = 0
            self.cuda_device_name = None
            self.cuda_memory_gb = 0
        
        # Determine device priority: CUDA > MPS > CPU
        if self.has_cuda:
            self.recommended_device = "cuda"
        elif self.has_mps:
            self.recommended_device = "mps"
        else:
            self.recommended_device = "cpu"
    
    def __repr__(self):
        lines = [
            "Machine Specifications:",
            f"  Platform: {self.platform} {self.platform_release}",
            f"  CPU: {self.physical_cores} physical cores, {self.logical_cores} logical cores",
            f"  RAM: {self.ram_gb:.2f} GB total, {self.available_ram_gb:.2f} GB available",
            f"  GPU (CUDA): {'Yes' if self.has_cuda else 'No'}",
        ]
        
        if self.has_cuda:
            lines.append(f"    Device: {self.cuda_device_name}")
            lines.append(f"    VRAM: {self.cuda_memory_gb:.2f} GB")
        
        lines.append(f"  GPU (MPS/Apple Silicon): {'Yes' if self.has_mps else 'No'}")
        lines.append(f"  Recommended Device: {self.recommended_device}")
        
        return "\n".join(lines)


class AutoConfig:
    """Automatically configures settings based on hardware specifications."""
    
    # Performance tiers based on hardware capabilities
    TIER_LOW = "low"
    TIER_MEDIUM = "medium"
    TIER_HIGH = "high"
    TIER_ULTRA = "ultra"
    
    def __init__(self, specs: MachineSpecs = None, upscale_factor: int = 2):
        self.specs = specs or MachineSpecs()
        self.upscale_factor = upscale_factor
        self.tier = self._determine_tier()
        logger.info(f"Hardware tier: {self.tier.upper()}")
    
    def _determine_tier(self) -> str:
        """Determine hardware performance tier."""
        ram = self.specs.ram_gb
        cores = self.specs.physical_cores
        has_gpu = self.specs.has_cuda or self.specs.has_mps
        
        # Ultra tier: High-end GPU + lots of RAM
        if self.specs.has_cuda and self.specs.cuda_memory_gb >= 8 and ram >= 16:
            return self.TIER_ULTRA
        
        # High tier: Any GPU + decent RAM + multi-core
        if has_gpu and ram >= 8 and cores >= 4:
            return self.TIER_HIGH
        
        # Medium tier: GPU or decent CPU + moderate RAM
        if (has_gpu and ram >= 4) or (cores >= 4 and ram >= 8):
            return self.TIER_MEDIUM
        
        # Low tier: Limited resources
        return self.TIER_LOW
    
    def get_optimal_threads(self) -> int:
        """Calculate optimal number of worker threads for DataLoader.
        
        Returns:
            Number of worker threads (0 for MPS to avoid multiprocessing issues)
        """
        # MPS requires 0 workers to avoid multiprocessing deadlocks
        if self.specs.recommended_device == "mps":
            return 0
        
        # Use physical cores, leave some headroom for OS
        cores = self.specs.physical_cores
        
        if cores <= 2:
            return 0  # Single-threaded for very limited systems
        elif cores <= 4:
            return 2
        elif cores <= 8:
            return min(4, cores - 1)
        else:
            return min(8, cores - 2)  # Cap at 8 to avoid diminishing returns
    
    def get_optimal_batch_size(self) -> Tuple[int, int]:
        """Calculate optimal batch sizes for training and testing.
        
        Returns:
            Tuple of (train_batch_size, test_batch_size)
        """
        # Estimate memory requirements based on upscale factor
        # 2x: ~200MB per batch of 16
        # 3x: ~350MB per batch of 16
        # 4x: ~500MB per batch of 16
        memory_per_sample_mb = {
            2: 12.5,
            3: 22,
            4: 31,
            8: 125
        }.get(self.upscale_factor, 50)
        
        if self.specs.has_cuda:
            # CUDA: Base on VRAM
            available_memory_gb = self.specs.cuda_memory_gb * 0.7  # 70% safety margin
            if available_memory_gb >= 8:
                train_batch = 32
                test_batch = 16
            elif available_memory_gb >= 4:
                train_batch = 16
                test_batch = 8
            elif available_memory_gb >= 2:
                train_batch = 8
                test_batch = 4
            else:
                train_batch = 4
                test_batch = 2
        
        elif self.specs.has_mps:
            # MPS: Base on system RAM (shared memory architecture)
            available_memory_gb = self.specs.ram_gb * 0.4  # 40% for model+data
            if available_memory_gb >= 8:
                train_batch = 16
                test_batch = 8
            elif available_memory_gb >= 4:
                train_batch = 8
                test_batch = 4
            else:
                train_batch = 4
                test_batch = 2
        
        else:
            # CPU: Conservative batch sizes
            available_memory_gb = self.specs.ram_gb * 0.5
            if available_memory_gb >= 8:
                train_batch = 8
                test_batch = 4
            elif available_memory_gb >= 4:
                train_batch = 4
                test_batch = 2
            else:
                train_batch = 2
                test_batch = 1
        
        # Adjust based on upscale factor (higher factors need more memory)
        if self.upscale_factor >= 4:
            train_batch = max(1, train_batch // 2)
            test_batch = max(1, test_batch // 2)
        
        return train_batch, test_batch
    
    def get_device_settings(self) -> Dict:
        """Get device-specific settings (CUDA/MPS/CPU)."""
        settings = {
            'cuda': False,
            'mps': False,
            'mixed_precision': False,
            'channels_last': False,
            'compile_model': False,
            'compile_mode': 'default',
            'use_fused_optimizer': False,
            'persistent_workers': False,
            'cache_dataset': False
        }
        
        if self.specs.recommended_device == "cuda":
            settings['cuda'] = True
            settings['mixed_precision'] = True  # FP16 on CUDA
            settings['channels_last'] = True
            
            # torch.compile on CUDA 11.7+ / PyTorch 2.0+
            if hasattr(torch, 'compile'):
                settings['compile_model'] = True
                # Ultra tier can use max-autotune for best performance
                if self.tier == self.TIER_ULTRA:
                    settings['compile_mode'] = 'max-autotune'
                else:
                    settings['compile_mode'] = 'default'
            
            # Fused optimizers on CUDA
            settings['use_fused_optimizer'] = True
            settings['persistent_workers'] = True
            
            # Cache dataset if lots of RAM
            if self.specs.ram_gb >= 16:
                settings['cache_dataset'] = True
        
        elif self.specs.recommended_device == "mps":
            settings['mps'] = True
            settings['channels_last'] = True  # Apple Silicon benefits from this
            # MPS doesn't support FP16 GradScaler, but other optimizations help
            settings['mixed_precision'] = False
            settings['compile_model'] = False  # torch.compile not stable on MPS yet
            settings['persistent_workers'] = False  # 0 workers on MPS
            
            # Cache dataset if sufficient RAM
            if self.specs.ram_gb >= 12:
                settings['cache_dataset'] = True
        
        else:  # CPU
            settings['channels_last'] = False
            settings['compile_model'] = False
            settings['persistent_workers'] = self.specs.physical_cores > 4
            
            # Only cache on CPU if lots of RAM available
            if self.specs.ram_gb >= 16:
                settings['cache_dataset'] = True
        
        return settings
    
    def get_training_settings(self) -> Dict:
        """Get optimized training settings."""
        train_batch, test_batch = self.get_optimal_batch_size()
        
        settings = {
            'batch_size': train_batch,
            'test_batch_size': test_batch,
            'threads': self.get_optimal_threads(),
            'gradient_accumulation_steps': 1,
        }
        
        # Use gradient accumulation on low-tier hardware to simulate larger batches
        if self.tier == self.TIER_LOW:
            settings['gradient_accumulation_steps'] = 2
        elif self.tier == self.TIER_ULTRA:
            # Ultra tier can handle larger effective batch sizes
            settings['gradient_accumulation_steps'] = 1
        
        return settings
    
    def get_optimizer_settings(self) -> Dict:
        """Get optimized optimizer settings based on hardware."""
        settings = {
            'optimizer': 'adam',  # Adam is generally robust
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'scheduler': True  # OneCycleLR works well
        }
        
        # AdamW can be better on high-tier hardware
        if self.tier in [self.TIER_HIGH, self.TIER_ULTRA]:
            settings['optimizer'] = 'adamw'
        
        return settings
    
    def generate_config(self, base_config_path: str = "settings.yaml") -> Dict:
        """Generate complete optimized configuration.
        
        Args:
            base_config_path: Path to existing settings.yaml to use as base
        
        Returns:
            Dictionary with optimized settings
        """
        # Load existing config
        try:
            with open(base_config_path, 'r') as f:
                config = yaml.load(f, Loader=Loader)
        except FileNotFoundError:
            logger.warning(f"Base config not found: {base_config_path}, using defaults")
            config = {}
        
        # Apply device settings
        config.update(self.get_device_settings())
        
        # Apply training settings
        config.update(self.get_training_settings())
        
        # Apply optimizer settings (only if not tuning)
        if not config.get('tuning', False):
            config.update(self.get_optimizer_settings())
        
        return config
    
    def save_config(self, output_path: str = "settings_auto.yaml"):
        """Generate and save optimized configuration to YAML file."""
        config = self.generate_config()
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, Dumper=Dumper, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Auto-configured settings saved to: {output_path}")
        return config
    
    def apply_to_current_settings(self, settings_path: str = "settings.yaml", backup: bool = True):
        """Apply autoconfiguration to current settings.yaml file.
        
        Args:
            settings_path: Path to settings.yaml
            backup: If True, create backup before modifying
        """
        if backup:
            import shutil
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"settings.yaml.backup_{timestamp}"
            shutil.copy(settings_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
        
        # Load current settings
        with open(settings_path, 'r') as f:
            current_config = yaml.load(f, Loader=Loader)
        
        # Generate optimizations
        optimized = self.generate_config(settings_path)
        
        # Merge (preserve user settings for non-hardware-related options)
        preserve_keys = [
            'input_path', 'output_path', 'model_path', 'upscale_factor',
            'mode', 'epochs_number', 'epoch', 'checkpoint_frequency',
            'preload', 'preload_path', 'seed', 'show_progress_bar',
            'show_profiler', 'show_result', 'verbose_logging', 'cycles',
            'tuning', 'trials', 'early_stopping', 'early_stopping_patience',
            'psnr_delta', 'stuck_level', 'target_min_psnr',
            'separable', 'dropout_rate', 'use_bn', 'optimized', 'num_classes',
            'pruning', 'prune_amount'
        ]
        
        for key in preserve_keys:
            if key in current_config:
                optimized[key] = current_config[key]
        
        # Save optimized settings
        with open(settings_path, 'w') as f:
            yaml.dump(optimized, f, Dumper=Dumper, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Auto-configured settings applied to: {settings_path}")
        return optimized
    
    def print_recommendations(self):
        """Print hardware specs and recommended settings."""
        print("\n" + "=" * 70)
        print("ESPCN AUTO-CONFIGURATION")
        print("=" * 70)
        print(self.specs)
        print("\nRecommended Settings:")
        print(f"  Performance Tier: {self.tier.upper()}")
        
        device_settings = self.get_device_settings()
        print(f"  Device: {self.specs.recommended_device.upper()}")
        print(f"  Mixed Precision: {device_settings['mixed_precision']}")
        print(f"  Channels Last: {device_settings['channels_last']}")
        print(f"  Compile Model: {device_settings['compile_model']}")
        if device_settings['compile_model']:
            print(f"    Compile Mode: {device_settings['compile_mode']}")
        
        train_settings = self.get_training_settings()
        print(f"\n  Batch Size (Train): {train_settings['batch_size']}")
        print(f"  Batch Size (Test): {train_settings['test_batch_size']}")
        print(f"  DataLoader Threads: {train_settings['threads']}")
        print(f"  Gradient Accumulation: {train_settings['gradient_accumulation_steps']}")
        
        print(f"\n  Cache Dataset: {device_settings['cache_dataset']}")
        print(f"  Persistent Workers: {device_settings['persistent_workers']}")
        print(f"  Fused Optimizer: {device_settings['use_fused_optimizer']}")
        
        print("=" * 70 + "\n")


def detect_and_configure(
    apply: bool = False,
    save_to: str = None,
    print_info: bool = True,
    upscale_factor: int = None
) -> Dict:
    """Main entry point for autoconfiguration.
    
    Args:
        apply: If True, applies changes to settings.yaml
        save_to: If provided, saves config to this file instead
        print_info: If True, prints recommendations
        upscale_factor: Upscale factor to optimize for (2, 3, 4, or 8).
                       If None, reads from current settings.yaml
    
    Returns:
        Dictionary with optimized settings
    """
    # Determine upscale factor
    if upscale_factor is None:
        try:
            with open("settings.yaml", 'r') as f:
                current_config = yaml.load(f, Loader=Loader)
                upscale_factor = current_config.get('upscale_factor', 2)
        except FileNotFoundError:
            upscale_factor = 2
            logger.warning("settings.yaml not found, using default upscale_factor=2")
    
    # Create autoconfig
    specs = MachineSpecs()
    autoconfig = AutoConfig(specs, upscale_factor=upscale_factor)
    
    if print_info:
        autoconfig.print_recommendations()
    
    if apply:
        return autoconfig.apply_to_current_settings()
    elif save_to:
        return autoconfig.save_config(save_to)
    else:
        return autoconfig.generate_config()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Auto-configure ESPCN settings based on hardware specs"
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply autoconfiguration to settings.yaml'
    )
    parser.add_argument(
        '--save-to',
        type=str,
        help='Save configuration to specified file'
    )
    parser.add_argument(
        '--upscale-factor',
        type=int,
        choices=[2, 3, 4, 8],
        help='Upscale factor to optimize for'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )
    
    args = parser.parse_args()
    
    detect_and_configure(
        apply=args.apply,
        save_to=args.save_to,
        print_info=not args.quiet,
        upscale_factor=args.upscale_factor
    )
