# ESPCN PyTorch Web Server

A modern web interface for training, evaluating, and monitoring ESPCN (Efficient Sub-Pixel Convolutional Neural Network) models for image super-resolution.

## Features

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

### Command Line Options

```bash
python web_server.py --host 0.0.0.0 --port 5000 --debug
```

- `--host`: Host to bind to (default: `0.0.0.0`)
- `--port`: Port to bind to (default: `5000`)
- `--debug`: Enable Flask debug mode

## API Endpoints

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

## Screenshots

The web interface features a modern dark theme with:
- Sidebar navigation
- Responsive grid layouts
- Interactive gauges and charts
- Toast notifications
- Lightbox image viewer

## License

MIT License - See main project LICENSE for details.
