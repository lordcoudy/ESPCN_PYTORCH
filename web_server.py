"""
ESPCN PyTorch Web Server
A Flask-based web interface for training, evaluating, and monitoring ESPCN models.
"""
import atexit
import json
import multiprocessing
import os
import platform
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from functools import wraps
from pathlib import Path

# Suppress multiprocessing semaphore warnings on macOS
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

# Fix multiprocessing on macOS - must use 'spawn' for compatibility
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import psutil
import yaml
from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory)
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__, 
            template_folder='web/templates',
            static_folder='web/static')
CORS(app)

# Configuration
SETTINGS_FILE = 'settings.yaml'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# Global state for training
training_state = {
    'is_running': False,
    'is_paused': False,
    'mode': None,  # 'train', 'demo', 'tune'
    'process': None,
    'start_time': None,
    'elapsed': 0.0,
    'progress': {
        'current_epoch': 0,
        'total_epochs': 0,
        'current_loss': 0,
        'psnr': 0,
        'best_psnr': 0,
        'eta': None,
        'message': 'Idle'
    },
    'logs': []
}

log_queue = queue.Queue(maxsize=1000)


def _safe_relative_path(base: Path, target: Path) -> Path:
    """Ensure target stays within base directory."""
    resolved_base = base.resolve()
    resolved_target = target.resolve()
    if resolved_base not in resolved_target.parents and resolved_base != resolved_target:
        raise ValueError("Invalid path outside allowed directory")
    return resolved_target


def _current_elapsed_time() -> float:
    """Return elapsed time, frozen when not running."""
    elapsed = training_state.get('elapsed', 0.0) or 0.0
    if training_state.get('is_running') and training_state.get('start_time'):
        elapsed += max(0.0, time.time() - training_state['start_time'])
    return elapsed


# ============== Utility Functions ==============

def load_settings():
    """Load settings from YAML file."""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {'error': str(e)}


def save_settings(settings):
    """Save settings to YAML file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        return False


def get_system_stats():
    """Get comprehensive system statistics."""
    stats = {
        'cpu': {
            'percent': psutil.cpu_percent(interval=0.1),
            'count': psutil.cpu_count(),
            'freq': None,
            'per_cpu': psutil.cpu_percent(interval=0.1, percpu=True)
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'used': psutil.virtual_memory().used,
            'percent': psutil.virtual_memory().percent,
            'available': psutil.virtual_memory().available
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'used': psutil.disk_usage('/').used,
            'percent': psutil.disk_usage('/').percent,
            'free': psutil.disk_usage('/').free
        },
        'system': {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'hostname': platform.node()
        },
        'gpu': get_gpu_stats()
    }
    
    # CPU frequency (may not be available on all systems)
    try:
        freq = psutil.cpu_freq()
        if freq:
            stats['cpu']['freq'] = {
                'current': freq.current,
                'min': freq.min,
                'max': freq.max
            }
    except:
        pass
    
    return stats


def get_gpu_stats():
    """Get GPU statistics (supports CUDA and Apple MPS)."""
    gpu_info = {
        'available': False,
        'type': None,
        'devices': []
    }
    
    try:
        import torch

        # Check CUDA
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['type'] = 'CUDA'
            for i in range(torch.cuda.device_count()):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_cached': torch.cuda.memory_reserved(i)
                }
                gpu_info['devices'].append(device_info)
        
        # Check MPS (Apple Silicon)
        elif torch.backends.mps.is_available():
            gpu_info['available'] = True
            gpu_info['type'] = 'MPS (Apple Silicon)'
            # MPS doesn't have detailed memory stats like CUDA
            gpu_info['devices'].append({
                'id': 0,
                'name': 'Apple Neural Engine',
                'memory_total': None,
                'memory_allocated': None
            })
    except ImportError:
        pass
    except Exception as e:
        gpu_info['error'] = str(e)
    
    return gpu_info


def get_available_models():
    """List all available trained models with only the latest checkpoint."""
    models = []
    models_path = Path(MODELS_DIR)
    
    if models_path.exists():
        for model_dir in models_path.iterdir():
            if model_dir.is_dir():
                checkpoints = list(model_dir.glob('*.pth'))
                # Get only the latest checkpoint (most recently modified)
                latest_checkpoint = None
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                
                model_info = {
                    'name': model_dir.name,
                    'path': str(model_dir),
                    'checkpoints': [
                        {
                            'name': latest_checkpoint.name,
                            'path': str(latest_checkpoint),
                            'size': latest_checkpoint.stat().st_size,
                            'modified': datetime.fromtimestamp(latest_checkpoint.stat().st_mtime).isoformat()
                        }
                    ] if latest_checkpoint else [],
                    'has_logs': (model_dir / 'logs').exists(),
                    'has_times': (model_dir / 'times').exists()
                }
                models.append(model_info)
    
    return sorted(models, key=lambda x: x['name'], reverse=True)


def get_results():
    """List all result images."""
    results = []
    results_path = Path(RESULTS_DIR)
    
    if results_path.exists():
        for result_dir in results_path.iterdir():
            if result_dir.is_dir():
                images = list(result_dir.glob('*.png')) + list(result_dir.glob('*.jpg'))
                result_info = {
                    'name': result_dir.name,
                    'path': str(result_dir),
                    'images': [
                        {
                            'name': img.name,
                            'path': str(img),
                            'size': img.stat().st_size,
                            'modified': datetime.fromtimestamp(img.stat().st_mtime).isoformat()
                        }
                        for img in sorted(images, key=lambda x: x.name)
                    ]
                }
                results.append(result_info)
    
    return sorted(results, key=lambda x: x['name'], reverse=True)


def run_training_process(mode='train'):
    """Run training/demo in a subprocess and capture output."""
    global training_state
    
    training_state['is_running'] = True
    training_state['elapsed'] = 0.0
    training_state['start_time'] = time.time()
    training_state['progress']['current_epoch'] = 0
    training_state['progress']['psnr'] = 0
    training_state['progress']['best_psnr'] = 0
    training_state['progress']['current_loss'] = 0
    training_state['logs'] = []

    try:
        # Set mode in settings and get epochs for progress display
        settings = load_settings()
        tuning_enabled = settings.get('tuning', False)
        trials = settings.get('trials', 0)
        # Always save mode as 'train' when launching tuning so main.py runs both stages
        mode_to_save = 'train' if mode in ('train', 'tune') else mode
        settings['mode'] = mode_to_save
        save_settings(settings)

        effective_label = 'tune' if tuning_enabled and mode_to_save == 'train' else mode_to_save
        training_state['mode'] = effective_label
        training_state['progress']['message'] = 'Starting tuning...' if effective_label == 'tune' else f'Starting {mode_to_save}...'

        # Initialize total_epochs for UI progress bar
        if effective_label == 'tune' and trials:
            training_state['progress']['total_epochs'] = trials
        elif effective_label == 'demo':
            def count_demo_items(cfg):
                input_path = cfg.get('input_path') or ''
                cycles = cfg.get('cycles', 1)
                if os.path.isfile(input_path):
                    return 1
                search_dir = input_path if os.path.isdir(input_path) else './dataset/BSDS500/images/test/'
                images = [f for f in os.listdir(search_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                return min(len(images), cycles) if images else 0

            training_state['progress']['total_epochs'] = count_demo_items(settings)
        else:
            training_state['progress']['total_epochs'] = settings.get('epochs_number', settings.get('epoch', 0))
        
        # Run the main script with unbuffered output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        process = subprocess.Popen(
            [sys.executable, '-u', 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
            env=env
        )
        training_state['process'] = process
        
        # Log that training has started
        training_state['logs'].append({
            'timestamp': datetime.now().isoformat(),
            'message': f'Process started (PID: {process.pid})'
        })
        
        # Read output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'message': line.strip()
                }
                training_state['logs'].append(log_entry)
                
                # Parse progress from log output
                parse_training_log(line)
                
                # Keep only last 500 log entries
                if len(training_state['logs']) > 500:
                    training_state['logs'] = training_state['logs'][-500:]
        
        process.wait()
        
    except Exception as e:
        training_state['logs'].append({
            'timestamp': datetime.now().isoformat(),
            'message': f'Error: {str(e)}'
        })
    finally:
        training_state['is_running'] = False
        # Freeze elapsed time when process ends
        training_state['elapsed'] = _current_elapsed_time()
        training_state['start_time'] = None
        training_state['process'] = None
        training_state['progress']['message'] = 'Completed' if training_state.get('mode') != 'tune' else 'Tuning completed'


def parse_training_log(line):
    """Parse training log line to extract progress information."""
    global training_state
    import re
    
    line = line.strip()

    # Track stage transitions
    if 'Tuning mode' in line:
        training_state['mode'] = 'tune'
        training_state['progress']['message'] = 'Tuning in progress...'
    if 'Training mode' in line:
        training_state['mode'] = 'train'
        training_state['progress']['current_epoch'] = 0
        training_state['progress']['psnr'] = 0
        training_state['progress']['best_psnr'] = 0
        training_state['progress']['message'] = 'Training in progress...'
    if 'Demo mode' in line:
        training_state['mode'] = 'demo'
        training_state['progress']['message'] = 'Demo in progress...'

    # Parse tuning progress: "Tuning trial X/Y"
    tune_match = re.search(r'Tuning\s+trial\s+(\d+)(?:/(\d+))?', line, re.IGNORECASE)
    if tune_match:
        trial_num = int(tune_match.group(1))
        training_state['progress']['current_epoch'] = trial_num
        if tune_match.group(2):
            training_state['progress']['total_epochs'] = int(tune_match.group(2))
        total_trials = training_state['progress'].get('total_epochs') or '?'
        training_state['progress']['message'] = f"Tuning trial {trial_num}/{total_trials}"

    # Parse demo progress: "Demo progress X/Y"
    demo_match = re.search(r'Demo\s+progress\s+(\d+)/(\d+)', line, re.IGNORECASE)
    if demo_match:
        current = int(demo_match.group(1))
        total = int(demo_match.group(2))
        training_state['mode'] = 'demo'
        training_state['progress']['current_epoch'] = current
        training_state['progress']['total_epochs'] = total
        training_state['progress']['message'] = f"Demo {current}/{total}"
    
    # Parse epoch information: "Epoch 1/2000 | Train Loss: 0.123... | Val Loss: 0.456..."
    epoch_match = re.search(r'Epoch\s+(\d+)/(\d+)\s*\|', line)
    if epoch_match:
        training_state['progress']['current_epoch'] = int(epoch_match.group(1))
        training_state['progress']['total_epochs'] = int(epoch_match.group(2))
    
    # Parse train loss from epoch line: "Train Loss: 0.000123456789"
    train_loss_match = re.search(r'Train\s+Loss:\s*([\d.e+-]+)', line)
    if train_loss_match:
        try:
            training_state['progress']['current_loss'] = float(train_loss_match.group(1))
        except ValueError:
            pass
    
    # Parse PSNR: "Avg. PSNR: 25.123456789012 dB"
    psnr_match = re.search(r'Avg\.\s*PSNR:\s*([\d.]+)\s*dB', line)
    if psnr_match:
        try:
            psnr = float(psnr_match.group(1))
            training_state['progress']['psnr'] = psnr
            if psnr > training_state['progress']['best_psnr']:
                training_state['progress']['best_psnr'] = psnr
        except ValueError:
            pass
    
    # Parse upscale factor and epochs from initial log: "Upscale factor: 2 | Epochs: 2000"
    init_match = re.search(r'Upscale\s+factor:\s*(\d+)\s*\|\s*Epochs:\s*(\d+)', line)
    if init_match:
        training_state['progress']['total_epochs'] = int(init_match.group(2))
        training_state['progress']['message'] = f"Initializing {init_match.group(1)}x model..."
    
    # Update message when training is in progress
    if training_state['progress']['current_epoch'] > 0:
        training_state['progress']['message'] = (
            f"Epoch {training_state['progress']['current_epoch']}/{training_state['progress']['total_epochs']} "
            f"- PSNR: {training_state['progress']['psnr']:.4f} dB"
        )


# ============== API Routes ==============

@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    """Get current settings."""
    return jsonify(load_settings())


@app.route('/api/settings', methods=['POST'])
def api_save_settings():
    """Save settings."""
    try:
        new_settings = request.json
        if save_settings(new_settings):
            return jsonify({'success': True, 'message': 'Settings saved successfully'})
        return jsonify({'success': False, 'message': 'Failed to save settings'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/settings/reset', methods=['POST'])
def api_reset_settings():
    """Reset settings singleton for hot-reload."""
    try:
        from settings import Settings
        Settings.reset()
        return jsonify({'success': True, 'message': 'Settings reset successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_system_stats():
    """Get system statistics."""
    return jsonify(get_system_stats())


@app.route('/api/stats/stream')
def api_stats_stream():
    """Server-sent events stream for real-time system stats."""
    def generate():
        while True:
            stats = get_system_stats()
            stats['training'] = {
                'is_running': training_state['is_running'],
                'mode': training_state['mode'],
                'progress': training_state['progress'],
                'elapsed_time': _current_elapsed_time()
            }
            yield f"data: {json.dumps(stats)}\n\n"
            time.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/models', methods=['GET'])
def api_get_models():
    """Get list of available models."""
    return jsonify(get_available_models())


@app.route('/api/models/delete', methods=['POST'])
def api_delete_model():
    """Delete a model directory and its contents."""
    try:
        model_name = request.json.get('name')
        if not model_name:
            return jsonify({'success': False, 'message': 'Model name required'}), 400
        target = _safe_relative_path(Path(MODELS_DIR), Path(MODELS_DIR) / model_name)
        if not target.exists():
            return jsonify({'success': False, 'message': 'Model not found'}), 404
        if not target.is_dir():
            return jsonify({'success': False, 'message': 'Invalid model path'}), 400
        shutil.rmtree(target)
        return jsonify({'success': True, 'message': 'Model deleted'})
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/results', methods=['GET'])
def api_get_results():
    """Get list of result images."""
    return jsonify(get_results())


@app.route('/api/results/delete', methods=['POST'])
def api_delete_result_set():
    """Delete a result directory."""
    try:
        result_name = request.json.get('name')
        if not result_name:
            return jsonify({'success': False, 'message': 'Result name required'}), 400
        target = _safe_relative_path(Path(RESULTS_DIR), Path(RESULTS_DIR) / result_name)
        if not target.exists():
            return jsonify({'success': False, 'message': 'Result set not found'}), 404
        if not target.is_dir():
            return jsonify({'success': False, 'message': 'Invalid result path'}), 400
        shutil.rmtree(target)
        return jsonify({'success': True, 'message': 'Result set deleted'})
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/results/<path:filepath>')
def api_serve_result(filepath):
    """Serve result image file."""
    return send_from_directory('.', filepath)


@app.route('/api/training/start', methods=['POST'])
def api_start_training():
    """Start training process."""
    global training_state
    
    if training_state['is_running']:
        return jsonify({'success': False, 'message': 'Training already in progress'}), 400

    payload = request.json or {}
    mode = payload.get('mode', 'train')
    tuning_requested = payload.get('tuning')
    trials_requested = payload.get('trials')

    settings = load_settings()
    if tuning_requested is not None:
        settings['tuning'] = bool(tuning_requested)
    if trials_requested is not None:
        try:
            settings['trials'] = int(trials_requested)
        except (TypeError, ValueError):
            pass

    # Ensure main.py runs training path even when tuning is enabled
    settings['mode'] = 'train' if mode in ('train', 'tune') else mode
    save_settings(settings)

    effective_mode = 'tune' if settings.get('tuning') and settings['mode'] == 'train' else settings['mode']

    # Start training in background thread
    thread = threading.Thread(target=run_training_process, args=(effective_mode,), daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'message': f'{effective_mode.capitalize()} started'})


@app.route('/api/training/stop', methods=['POST'])
def api_stop_training():
    """Stop training process."""
    global training_state
    
    if not training_state['is_running']:
        return jsonify({'success': False, 'message': 'No training in progress'}), 400
    
    if training_state['process']:
        training_state['process'].terminate()
        training_state['elapsed'] = _current_elapsed_time()
        training_state['start_time'] = None
        training_state['is_running'] = False
        training_state['progress']['message'] = 'Stopped by user'
    
    return jsonify({'success': True, 'message': 'Training stopped'})


@app.route('/api/training/status', methods=['GET'])
def api_training_status():
    """Get current training status."""
    return jsonify({
        'is_running': training_state['is_running'],
        'is_paused': training_state['is_paused'],
        'mode': training_state['mode'],
        'progress': training_state['progress'],
        'elapsed_time': _current_elapsed_time()
    })


@app.route('/api/training/logs', methods=['GET'])
def api_training_logs():
    """Get training logs."""
    limit = request.args.get('limit', 100, type=int)
    return jsonify(training_state['logs'][-limit:])


@app.route('/api/training/logs/stream')
def api_logs_stream():
    """Server-sent events stream for real-time logs."""
    def generate():
        last_count = 0
        while True:
            current_count = len(training_state['logs'])
            if current_count > last_count:
                new_logs = training_state['logs'][last_count:]
                for log in new_logs:
                    yield f"data: {json.dumps(log)}\n\n"
                last_count = current_count
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/demo/run', methods=['POST'])
def api_run_demo():
    """Run demo/evaluation on images."""
    global training_state
    
    if training_state['is_running']:
        return jsonify({'success': False, 'message': 'Another process is running'}), 400
    
    # Optionally override input path
    input_path = request.json.get('input_path')
    if input_path:
        settings = load_settings()
        settings['input_path'] = input_path
        settings['mode'] = 'demo'
        save_settings(settings)
    
    # Start demo in background thread
    thread = threading.Thread(target=run_training_process, args=('demo',), daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Demo started'})


@app.route('/api/tune/start', methods=['POST'])
def api_start_tuning():
    """Start hyperparameter tuning."""
    global training_state
    
    if training_state['is_running']:
        return jsonify({'success': False, 'message': 'Another process is running'}), 400
    
    # Enable tuning in settings
    settings = load_settings()
    settings['tuning'] = True
    settings['mode'] = 'train'
    save_settings(settings)
    
    # Start tuning in background thread
    thread = threading.Thread(target=run_training_process, args=('tune',), daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'message': 'Hyperparameter tuning started'})


@app.route('/api/model/load', methods=['POST'])
def api_load_model():
    """Update settings to preload a specific model."""
    model_path = request.json.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'success': False, 'message': 'Invalid model path'}), 400
    
    settings = load_settings()
    settings['preload'] = True
    settings['preload_path'] = model_path
    save_settings(settings)
    
    return jsonify({'success': True, 'message': f'Model path set to {model_path}'})


@app.route('/api/datasets', methods=['GET'])
def api_get_datasets():
    """Get available datasets."""
    datasets = []
    dataset_path = Path('dataset')
    
    if dataset_path.exists():
        for ds in dataset_path.iterdir():
            if ds.is_dir():
                datasets.append({
                    'name': ds.name,
                    'path': str(ds),
                    'has_train': (ds / 'images' / 'train').exists(),
                    'has_test': (ds / 'images' / 'test').exists(),
                    'has_val': (ds / 'images' / 'val').exists()
                })
    
    return jsonify(datasets)


@app.route('/api/logs/model/<path:model_name>', methods=['GET'])
def api_get_model_logs(model_name):
    """Get logs for a specific model."""
    logs_path = Path(MODELS_DIR) / model_name / 'logs'
    times_path = Path(MODELS_DIR) / model_name / 'times'
    psnr_path = Path(MODELS_DIR) / model_name / 'max_psnrs.txt'
    
    result = {
        'logs': [],
        'times': [],
        'psnrs': []
    }
    
    if logs_path.exists():
        for log_file in logs_path.glob('*.log'):
            result['logs'].append({
                'name': log_file.name,
                'content': log_file.read_text()
            })
    
    if times_path.exists():
        for time_file in times_path.glob('*.txt'):
            result['times'].append({
                'name': time_file.name,
                'content': time_file.read_text()
            })
    
    if psnr_path.exists():
        result['psnrs'] = psnr_path.read_text().strip().split('\n')
    
    return jsonify(result)


# ============== AutoConfig Routes ==============

@app.route('/api/autoconfig/detect', methods=['GET'])
def api_autoconfig_detect():
    """Detect hardware and get recommendations."""
    try:
        from autoconfig import AutoConfig, MachineSpecs
        
        upscale_factor = request.args.get('upscale_factor', 2, type=int)
        
        specs = MachineSpecs()
        autoconfig = AutoConfig(specs, upscale_factor=upscale_factor)
        
        # Format hardware specs
        hw_info = {
            'platform': specs.platform,
            'platform_release': specs.platform_release,
            'cpu_cores': {
                'physical': specs.physical_cores,
                'logical': specs.logical_cores
            },
            'ram_gb': {
                'total': round(specs.ram_gb, 2),
                'available': round(specs.available_ram_gb, 2)
            },
            'gpu': {
                'has_cuda': specs.has_cuda,
                'has_mps': specs.has_mps,
                'recommended_device': specs.recommended_device
            }
        }
        
        if specs.has_cuda:
            hw_info['gpu']['cuda_device_count'] = specs.cuda_device_count
            hw_info['gpu']['cuda_device_name'] = specs.cuda_device_name
            hw_info['gpu']['cuda_memory_gb'] = round(specs.cuda_memory_gb, 2) if specs.cuda_memory_gb > 0 else 0
        
        # Get tier and recommendations
        recommendations = {
            'tier': autoconfig.tier.upper(),
            'device_settings': autoconfig.get_device_settings(),
            'training_settings': autoconfig.get_training_settings(),
            'optimizer_settings': autoconfig.get_optimizer_settings(),
            'complete_config': autoconfig.generate_config(SETTINGS_FILE)
        }
        
        return jsonify({
            'success': True,
            'hardware': hw_info,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/autoconfig/apply', methods=['POST'])
def api_autoconfig_apply():
    """Apply autoconfiguration to settings."""
    try:
        from autoconfig import AutoConfig, MachineSpecs
        
        upscale_factor = request.json.get('upscale_factor')
        create_backup = request.json.get('backup', True)
        
        # Load current settings for upscale factor if not specified
        if upscale_factor is None:
            current = load_settings()
            upscale_factor = current.get('upscale_factor', 2)
        
        # Generate and apply configuration
        specs = MachineSpecs()
        autoconfig = AutoConfig(specs, upscale_factor=upscale_factor)
        optimized_config = autoconfig.apply_to_current_settings(SETTINGS_FILE, backup=create_backup)
        
        return jsonify({
            'success': True,
            'message': 'Autoconfiguration applied successfully',
            'config': optimized_config,
            'tier': autoconfig.tier.upper()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/autoconfig/compare', methods=['GET'])
def api_autoconfig_compare():
    """Compare current settings with recommendations."""
    try:
        from autoconfig import AutoConfig, MachineSpecs
        
        current = load_settings()
        upscale_factor = current.get('upscale_factor', 2)
        
        specs = MachineSpecs()
        autoconfig = AutoConfig(specs, upscale_factor=upscale_factor)
        recommended = autoconfig.generate_config(SETTINGS_FILE)
        
        # Group comparisons
        categories = {
            'device': ['cuda', 'mps', 'mixed_precision', 'channels_last', 'compile_model', 'compile_mode'],
            'performance': ['batch_size', 'test_batch_size', 'threads', 'gradient_accumulation_steps', 
                          'persistent_workers', 'cache_dataset', 'use_fused_optimizer'],
            'optimizer': ['optimizer', 'learning_rate']
        }
        
        comparison = {}
        for category, keys in categories.items():
            comparison[category] = {}
            for key in keys:
                comparison[category][key] = {
                    'current': current.get(key),
                    'recommended': recommended.get(key),
                    'matches': current.get(key) == recommended.get(key)
                }
        
        return jsonify({
            'success': True,
            'tier': autoconfig.tier.upper(),
            'comparison': comparison
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def cleanup():
    """Clean up resources on exit."""
    global training_state
    if training_state['process'] is not None:
        try:
            training_state['process'].terminate()
            training_state['process'].wait(timeout=5)
        except Exception:
            pass
    training_state['is_running'] = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask development server."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          ESPCN PyTorch Web Server                            ║
╠══════════════════════════════════════════════════════════════╣
║  Dashboard: http://{host}:{port}                               ║
║  API Docs:  http://{host}:{port}/api                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ESPCN PyTorch Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, debug=args.debug)
