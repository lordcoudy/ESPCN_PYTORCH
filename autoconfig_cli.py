#!/usr/bin/env python3
"""
Quick hardware detection and settings autoconfiguration utility.

Usage:
    python autoconfig_cli.py              # Show recommendations only
    python autoconfig_cli.py --apply      # Apply to settings.yaml
    python autoconfig_cli.py --compare    # Compare current vs recommended
"""

import sys
from pathlib import Path

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from autoconfig import AutoConfig, MachineSpecs, detect_and_configure


def compare_settings(current_path: str = "settings.yaml"):
    """Compare current settings with recommendations."""
    print("\n" + "=" * 70)
    print("SETTINGS COMPARISON: Current vs Recommended")
    print("=" * 70)
    
    # Load current settings
    try:
        with open(current_path, 'r') as f:
            current = yaml.load(f, Loader=Loader)
    except FileNotFoundError:
        print(f"Error: {current_path} not found")
        return
    
    # Get recommendations
    specs = MachineSpecs()
    upscale_factor = current.get('upscale_factor', 2)
    autoconfig = AutoConfig(specs, upscale_factor=upscale_factor)
    recommended = autoconfig.generate_config(current_path)
    
    # Compare key settings
    compare_keys = [
        ('Device Settings', [
            ('cuda', 'CUDA'),
            ('mps', 'MPS'),
            ('mixed_precision', 'Mixed Precision'),
            ('channels_last', 'Channels Last'),
            ('compile_model', 'Compile Model'),
            ('compile_mode', 'Compile Mode')
        ]),
        ('Performance', [
            ('batch_size', 'Batch Size'),
            ('test_batch_size', 'Test Batch Size'),
            ('threads', 'Worker Threads'),
            ('gradient_accumulation_steps', 'Gradient Accumulation'),
            ('persistent_workers', 'Persistent Workers'),
            ('cache_dataset', 'Cache Dataset')
        ]),
        ('Optimizer', [
            ('optimizer', 'Optimizer'),
            ('learning_rate', 'Learning Rate'),
            ('use_fused_optimizer', 'Fused Optimizer')
        ])
    ]
    
    print(f"\nHardware Tier: {autoconfig.tier.upper()}")
    print(f"Recommended Device: {specs.recommended_device.upper()}")
    
    for section_name, keys in compare_keys:
        print(f"\n{section_name}:")
        print("-" * 70)
        
        for key, display_name in keys:
            current_val = current.get(key, 'N/A')
            recommended_val = recommended.get(key, 'N/A')
            
            # Format values
            if isinstance(current_val, float):
                current_str = f"{current_val:.6f}" if current_val < 0.01 else f"{current_val:.4f}"
                recommended_str = f"{recommended_val:.6f}" if recommended_val < 0.01 else f"{recommended_val:.4f}"
            else:
                current_str = str(current_val)
                recommended_str = str(recommended_val)
            
            # Highlight differences
            marker = "⚠️  " if current_val != recommended_val else "✓  "
            
            print(f"  {marker}{display_name:.<30} Current: {current_str:>12} | Recommended: {recommended_str:>12}")
    
    print("\n" + "=" * 70)
    print("Legend: ✓ = Matches recommendation, ⚠️  = Different from recommendation")
    print("=" * 70 + "\n")


def show_hardware_info():
    """Display detailed hardware information."""
    specs = MachineSpecs()
    print(specs)


def interactive_configure():
    """Interactive configuration wizard."""
    print("\n" + "=" * 70)
    print("ESPCN AUTOCONFIGURATION WIZARD")
    print("=" * 70)
    
    # Show hardware
    specs = MachineSpecs()
    print(specs)
    
    # Get upscale factor
    print("\nSelect upscale factor:")
    print("  1) 2x (fastest, least memory)")
    print("  2) 3x (balanced)")
    print("  3) 4x (slower, more memory)")
    print("  4) 8x (slowest, most memory)")
    
    try:
        choice = input("\nEnter choice [1-4] (default=1): ").strip()
        factor_map = {'1': 2, '2': 3, '3': 4, '4': 8, '': 2}
        upscale_factor = factor_map.get(choice, 2)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return
    
    # Create config
    autoconfig = AutoConfig(specs, upscale_factor=upscale_factor)
    
    print(f"\nGenerating optimized configuration for {upscale_factor}x upscaling...")
    print(f"Hardware tier: {autoconfig.tier.upper()}")
    
    # Ask to apply
    print("\nOptions:")
    print("  1) Show recommendations only")
    print("  2) Save to settings_auto.yaml")
    print("  3) Apply to settings.yaml (with backup)")
    
    try:
        choice = input("\nEnter choice [1-3] (default=1): ").strip()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return
    
    if choice == '1' or choice == '':
        autoconfig.print_recommendations()
    elif choice == '2':
        config = autoconfig.save_config("settings_auto.yaml")
        print("\n✓ Configuration saved to: settings_auto.yaml")
        print("  You can review it and manually copy to settings.yaml")
    elif choice == '3':
        config = autoconfig.apply_to_current_settings(backup=True)
        print("\n✓ Configuration applied to settings.yaml")
        print("  Backup created with timestamp")
        print("\nYou can now run training with optimized settings!")
    else:
        print("Invalid choice.")


def main():
    if len(sys.argv) == 1:
        # No arguments - show interactive wizard
        interactive_configure()
    
    elif '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --apply          Apply autoconfiguration to settings.yaml")
        print("  --compare        Compare current settings with recommendations")
        print("  --info           Show hardware information")
        print("  --interactive    Run interactive configuration wizard")
        print("  --save FILE      Save configuration to FILE")
        print("  --upscale N      Set upscale factor (2, 3, 4, or 8)")
        print("  --quiet          Suppress verbose output")
        print("  --help, -h       Show this help message")
    
    elif '--info' in sys.argv:
        show_hardware_info()
    
    elif '--compare' in sys.argv:
        compare_settings()
    
    elif '--interactive' in sys.argv:
        interactive_configure()
    
    else:
        # Parse arguments
        apply = '--apply' in sys.argv
        save_to = None
        quiet = '--quiet' in sys.argv
        upscale_factor = None
        
        if '--save' in sys.argv:
            idx = sys.argv.index('--save')
            if idx + 1 < len(sys.argv):
                save_to = sys.argv[idx + 1]
        
        if '--upscale' in sys.argv:
            idx = sys.argv.index('--upscale')
            if idx + 1 < len(sys.argv):
                try:
                    upscale_factor = int(sys.argv[idx + 1])
                    if upscale_factor not in [2, 3, 4, 8]:
                        print(f"Error: upscale factor must be 2, 3, 4, or 8 (got {upscale_factor})")
                        sys.exit(1)
                except ValueError:
                    print(f"Error: invalid upscale factor: {sys.argv[idx + 1]}")
                    sys.exit(1)
        
        # Run autoconfiguration
        detect_and_configure(
            apply=apply,
            save_to=save_to,
            print_info=not quiet,
            upscale_factor=upscale_factor
        )
        
        if apply:
            print("\n✓ Settings applied successfully!")
        elif save_to:
            print(f"\n✓ Settings saved to: {save_to}")


if __name__ == "__main__":
    main()
