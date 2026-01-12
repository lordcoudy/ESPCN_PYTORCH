# ESPCN PyTorch Web Server

A modern web interface for training, evaluating, and monitoring ESPCN (Efficient Sub-Pixel Convolutional Neural Network) models for image super-resolution.

## Features

### ⚡ **NEW** Auto-Configuration
- Automatic hardware detection (CPU, RAM, GPU)
- Performance tier classification (ULTRA/HIGH/MEDIUM/LOW)
- One-click optimization based on your hardware
- Settings comparison (current vs recommended)
- Safe application with automatic backups
- Preserves custom user settings

### 🖥️ Dashboard
- Real-time system monitoring (CPU, Memory, Disk, GPU)
- Training progress tracking with PSNR metrics
- Live log streaming
- Quick action buttons for training/demo/tuning

### 🏋️ Training Control
- Configure all training parameters via GUI
- Support for CUDA, MPS (Apple Silicon), and CPU
- Hyperparameter tuning with Optuna
- Model preloading for continued training

### ⚙️ Settings Management
- Visual settings editor
- YAML configuration editor
- Hot-reload settings without restart
- Advanced optimization options (mixed precision, compile, fused optimizer, etc.)

### 📦 Models Browser
- Browse all trained model checkpoints
- Load models for inference or continued training
- View training metadata (upscale factor, device, optimizations)

### 🖼️ Results Viewer
- Image gallery with lightbox
- Browse super-resolved images by result set
- Keyboard navigation support

### 📊 System Monitor
- Real-time CPU/Memory/Disk gauges
- Per-core CPU usage visualization
- GPU detection and stats (CUDA/MPS)
- System information display

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the web server:
```bash
python web_server.py
```

3. Open your browser at `http://localhost:5000`

### Using Auto-Configuration

1. Navigate to the **Auto-Config** tab in the sidebar
2. Click **Detect Hardware** to analyze your system
3. Review the detected hardware specs and performance tier
4. Check the **Recommended Settings** section
5. Optionally, click **Compare** to see differences from current settings
6. Select your desired upscale factor (2x, 3x, 4x, or 8x)
7. Click **Apply Auto-Configuration** to optimize your settings

**Features:**
- ✅ Detects CPU cores, RAM, and GPU (CUDA/MPS)
- ✅ Classifies hardware into performance tiers (ULTRA/HIGH/MEDIUM/LOW)
- ✅ Recommends optimal batch sizes, worker threads, and optimizations
- ✅ Preserves your custom settings (paths, epochs, seeds)
- ✅ Creates automatic backup before applying changes
- ✅ One-click optimization for best training performance

### Command Line Options

```bash
python web_server.py --host 0.0.0.0 --port 5000 --debug
```

- `--host`: Host to bind to (default: `0.0.0.0`)
- `--port`: Port to bind to (default: `5000`)
- `--debug`: Enable Flask debug mode

## API Endpoints

### ⚡ Auto-Configuration
- `GET /api/autoconfig/detect?upscale_factor=2` - Detect hardware and get recommendations
- `POST /api/autoconfig/apply` - Apply autoconfiguration to settings
- `GET /api/autoconfig/compare` - Compare current vs recommended settings

### Settings
- `GET /api/settings` - Get current settings
- `POST /api/settings` - Save settings
- `POST /api/settings/reset` - Reset settings singleton

### Training
- `POST /api/training/start` - Start training (`{"mode": "train"}`)
- `POST /api/training/stop` - Stop training
- `GET /api/training/status` - Get training status
- `GET /api/training/logs` - Get training logs
- `GET /api/training/logs/stream` - SSE stream for live logs

### Demo
- `POST /api/demo/run` - Run demo on images

### Tuning
- `POST /api/tune/start` - Start hyperparameter tuning

### Models
- `GET /api/models` - List all trained models
- `POST /api/model/load` - Set model for preloading

### Results
- `GET /api/results` - List all result sets
- `GET /api/results/<path>` - Serve result image

### System
- `GET /api/stats` - Get system statistics
- `GET /api/stats/stream` - SSE stream for live stats

### Datasets
- `GET /api/datasets` - List available datasets

## Technology Stack

- **Backend**: Flask, Flask-CORS
- **Frontend**: Vanilla JavaScript, CSS3
- **Monitoring**: psutil
- **Real-time Updates**: Server-Sent Events (SSE)
- **Auto-Configuration**: Hardware detection and optimization engine

## Auto-Configuration Guide

### What Does Auto-Config Do?

The Auto-Configuration system automatically detects your hardware and recommends optimal settings for training ESPCN models. It analyzes:

- **CPU**: Physical/logical cores for optimal worker threads
- **RAM**: Available memory for batch size optimization
- **GPU**: CUDA or Apple Silicon (MPS) detection
- **Storage**: Disk space for dataset caching decisions

### Performance Tiers

Your hardware is classified into one of four tiers:

| Tier | Characteristics | Example Hardware |
|------|-----------------|------------------|
| **ULTRA** | High-end GPU (8+ GB VRAM) + 16+ GB RAM | RTX 3090, 16+ cores, 32 GB RAM |
| **HIGH** | Good GPU + 8+ GB RAM + 4+ cores | RTX 2080, M1 Max, decent CPU/RAM |
| **MEDIUM** | Moderate resources | older GPU, 8 GB RAM, 4 cores |
| **LOW** | Limited resources | CPU-only or minimal GPU, <8 GB RAM |

### Recommended Settings by Tier

**ULTRA Tier:**
- Batch size: 32/16 (train/test)
- Worker threads: 8
- Mixed precision: Yes (FP16)
- Model compilation: max-autotune
- Fused optimizer: Yes
- Dataset caching: Yes

**HIGH Tier:**
- Batch size: 16/8
- Worker threads: 4-8
- Mixed precision: Yes (CUDA) or optimized (MPS)
- Model compilation: default
- Fused optimizer: Yes (CUDA)
- Dataset caching: Yes (if RAM sufficient)

**MEDIUM Tier:**
- Batch size: 8/4
- Worker threads: 2-4
- Mixed precision: No (memory efficient)
- Model compilation: No
- Gradient accumulation: 1
- Cache dataset: No

**LOW Tier:**
- Batch size: 4/2
- Worker threads: 0-2
- Gradient accumulation: 2x (simulates larger batches)
- Conservative memory usage
- All heavy optimizations disabled

### Settings Preservation

Auto-Config is intelligent about what it changes:

**Preserved (User Settings):**
- Input/output paths
- Epochs and training configuration
- Random seed
- Model architecture choices
- Early stopping parameters
- Tuning settings

**Optimized (Hardware-Specific):**
- Device selection (CUDA/MPS/CPU)
- Batch sizes
- Worker threads
- Memory optimizations
- Performance features

### Comparison View

Use the **Settings Comparison** to see:
- ✅ Which settings match recommendations
- ⚠️ Which settings differ from optimal
- Detailed side-by-side comparison

## Screenshots

The web interface features:
- Modern dark theme inspired by GitHub
- Sidebar navigation with quick access
- Responsive grid layouts
- Interactive gauges for system resources
- Real-time monitoring and training progress
- Advanced settings editor with YAML support
- Image gallery with lightbox viewer
- Auto-Config dashboard for hardware optimization

## License

MIT License - See main project LICENSE for details.
