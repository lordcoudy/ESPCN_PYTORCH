/**
 * ESPCN PyTorch Web UI - Main Application JavaScript
 */

// ============================================
// Global State
// ============================================
const state = {
    settings: {},
    models: [],
    results: [],
    currentResultSet: null,
    lightboxImages: [],
    lightboxIndex: 0,
    statsEventSource: null,
    logsEventSource: null
};

// ============================================
// Utility Functions
// ============================================
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0 || bytes === null) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    toast.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: { 'Content-Type': 'application/json' }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`/api${endpoint}`, options);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.message || 'API request failed');
        }
        
        return result;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ============================================
// Navigation
// ============================================
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Update active nav item
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            // Show corresponding tab
            const tabId = item.dataset.tab;
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById(`${tabId}-tab`).classList.add('active');
            
            // Load data for specific tabs
            if (tabId === 'models') loadModels();
            if (tabId === 'results') loadResults();
            if (tabId === 'settings') loadSettings();
        });
    });
}

// ============================================
// Dashboard & Stats
// ============================================
function initStatsStream() {
    if (state.statsEventSource) {
        state.statsEventSource.close();
    }
    
    state.statsEventSource = new EventSource('/api/stats/stream');
    
    state.statsEventSource.onmessage = (event) => {
        const stats = JSON.parse(event.data);
        updateStats(stats);
    };
    
    state.statsEventSource.onerror = () => {
        document.getElementById('server-status').classList.add('offline');
    };
}

function updateStats(stats) {
    // Update server status
    document.getElementById('server-status').classList.remove('offline');
    
    // CPU
    const cpuPercent = stats.cpu.percent.toFixed(1);
    document.getElementById('cpu-usage').textContent = cpuPercent + '%';
    document.getElementById('cpu-progress').style.width = cpuPercent + '%';
    document.getElementById('cpu-gauge-value').textContent = cpuPercent + '%';
    updateGauge('cpu-gauge', stats.cpu.percent);
    document.getElementById('cpu-count').textContent = stats.cpu.count;
    
    if (stats.cpu.freq) {
        document.getElementById('cpu-freq').textContent = 
            `${(stats.cpu.freq.current / 1000).toFixed(2)} GHz`;
    }
    
    // CPU cores
    const coresContainer = document.getElementById('cpu-cores');
    if (stats.cpu.per_cpu && coresContainer) {
        coresContainer.innerHTML = stats.cpu.per_cpu.map((usage, i) => `
            <div class="core-bar" title="Core ${i}: ${usage}%">
                <div class="core-bar-fill" style="height: ${usage}%"></div>
            </div>
        `).join('');
    }
    
    // Memory
    const memPercent = stats.memory.percent.toFixed(1);
    document.getElementById('memory-usage').textContent = memPercent + '%';
    document.getElementById('memory-progress').style.width = memPercent + '%';
    document.getElementById('memory-gauge-value').textContent = memPercent + '%';
    updateGauge('memory-gauge', stats.memory.percent);
    document.getElementById('memory-used').textContent = formatBytes(stats.memory.used);
    document.getElementById('memory-available').textContent = formatBytes(stats.memory.available);
    document.getElementById('memory-total').textContent = formatBytes(stats.memory.total);
    
    // Disk
    document.getElementById('disk-gauge-value').textContent = stats.disk.percent.toFixed(1) + '%';
    updateGauge('disk-gauge', stats.disk.percent);
    document.getElementById('disk-used').textContent = formatBytes(stats.disk.used);
    document.getElementById('disk-free').textContent = formatBytes(stats.disk.free);
    document.getElementById('disk-total').textContent = formatBytes(stats.disk.total);
    
    // GPU
    updateGPUInfo(stats.gpu);
    
    // Training Status
    if (stats.training) {
        updateTrainingStatus(stats.training);
    }
    
    // System Info
    updateSystemInfo(stats.system);
}

function updateGauge(gaugeId, value) {
    const gauge = document.getElementById(gaugeId);
    if (gauge) {
        const fill = gauge.querySelector('.gauge-fill');
        if (fill) {
            fill.style.setProperty('--value', value);
            fill.style.background = `conic-gradient(
                var(--accent-primary) ${value * 3.6}deg,
                var(--bg-tertiary) 0deg
            )`;
        }
    }
}

function updateGPUInfo(gpu) {
    const gpuInfo = document.getElementById('gpu-info');
    const gpuStatus = document.getElementById('gpu-status');
    const gpuProgress = document.getElementById('gpu-progress');
    
    if (gpu.available) {
        gpuStatus.textContent = gpu.type;
        gpuInfo.innerHTML = `
            <p class="gpu-available"><i class="fas fa-check-circle"></i> ${gpu.type} Available</p>
            ${gpu.devices.map(device => `
                <div class="system-details">
                    <p><strong>Device:</strong> <span>${device.name}</span></p>
                    ${device.memory_total ? `
                        <p><strong>Memory:</strong> <span>${formatBytes(device.memory_allocated)} / ${formatBytes(device.memory_total)}</span></p>
                    ` : ''}
                </div>
            `).join('')}
        `;
        
        if (gpu.devices[0]?.memory_total && gpu.devices[0]?.memory_allocated) {
            const memPercent = (gpu.devices[0].memory_allocated / gpu.devices[0].memory_total * 100);
            gpuProgress.style.width = memPercent + '%';
        }
    } else {
        gpuStatus.textContent = 'CPU Only';
        gpuInfo.innerHTML = `
            <p class="gpu-unavailable"><i class="fas fa-times-circle"></i> No GPU detected</p>
            <p style="font-size: 0.875rem; color: var(--text-muted);">Training will run on CPU</p>
        `;
        gpuProgress.style.width = '0%';
    }
}

function updateTrainingStatus(training) {
    const statusEl = document.getElementById('training-status');
    const startBtn = document.getElementById('btn-start-training');
    const stopBtn = document.getElementById('btn-stop-training');
    const startBtn2 = document.getElementById('btn-start-training-2');
    
    if (training.is_running) {
        statusEl.textContent = training.mode ? training.mode.charAt(0).toUpperCase() + training.mode.slice(1) : 'Running';
        statusEl.style.color = 'var(--accent-success)';
        startBtn.disabled = true;
        startBtn2.disabled = true;
        stopBtn.disabled = false;
    } else {
        statusEl.textContent = 'Idle';
        statusEl.style.color = '';
        startBtn.disabled = false;
        startBtn2.disabled = false;
        stopBtn.disabled = true;
    }
    
    // Update progress
    const progress = training.progress;
    document.getElementById('current-epoch').textContent = 
        `${progress.current_epoch} / ${progress.total_epochs || '-'}`;
    document.getElementById('current-psnr').textContent = 
        `${progress.psnr?.toFixed(4) || '0.0000'} dB`;
    document.getElementById('best-psnr').textContent = 
        `${progress.best_psnr?.toFixed(4) || '0.0000'} dB`;
    document.getElementById('elapsed-time').textContent = 
        formatTime(training.elapsed_time || 0);
    
    // Epoch progress bar
    if (progress.total_epochs > 0) {
        const percent = (progress.current_epoch / progress.total_epochs * 100).toFixed(1);
        document.getElementById('epoch-progress-bar').style.width = percent + '%';
        document.getElementById('epoch-progress-text').textContent = percent + '%';
    }
}

function updateSystemInfo(system) {
    const grid = document.getElementById('system-info-grid');
    if (grid && system) {
        grid.innerHTML = `
            <div class="info-item">
                <label>Platform</label>
                <span>${system.platform} ${system.platform_release}</span>
            </div>
            <div class="info-item">
                <label>Processor</label>
                <span>${system.processor || 'Unknown'}</span>
            </div>
            <div class="info-item">
                <label>Hostname</label>
                <span>${system.hostname}</span>
            </div>
            <div class="info-item">
                <label>Python Version</label>
                <span>${system.python_version.split(' ')[0]}</span>
            </div>
        `;
    }
}

// ============================================
// Training Controls
// ============================================
function initTrainingControls() {
    // Start Training button (Dashboard)
    document.getElementById('btn-start-training').addEventListener('click', async () => {
        try {
            await apiCall('/training/start', 'POST', { mode: 'train' });
            showToast('Training started!', 'success');
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
    
    // Start Training button (Training tab)
    document.getElementById('btn-start-training-2').addEventListener('click', async () => {
        await saveTrainingConfig();
        try {
            await apiCall('/training/start', 'POST', { mode: 'train' });
            showToast('Training started!', 'success');
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
    
    // Stop Training button
    document.getElementById('btn-stop-training').addEventListener('click', async () => {
        try {
            await apiCall('/training/stop', 'POST');
            showToast('Training stopped', 'warning');
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
    
    // Run Demo button
    document.getElementById('btn-run-demo').addEventListener('click', async () => {
        try {
            await apiCall('/demo/run', 'POST', {});
            showToast('Demo started!', 'success');
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
    
    // Tune button
    document.getElementById('btn-tune').addEventListener('click', async () => {
        try {
            await apiCall('/tune/start', 'POST');
            showToast('Hyperparameter tuning started!', 'success');
        } catch (error) {
            showToast(error.message, 'error');
        }
    });
    
    // Save Config button
    document.getElementById('btn-save-config').addEventListener('click', saveTrainingConfig);
    
    // Pruning toggle
    document.getElementById('train-pruning').addEventListener('change', (e) => {
        document.getElementById('prune-amount-group').style.display = 
            e.target.checked ? 'block' : 'none';
    });
    
    // Prune amount slider
    document.getElementById('train-prune-amount').addEventListener('input', (e) => {
        document.getElementById('prune-amount-value').textContent = e.target.value;
    });
    
    // Preload toggle
    document.getElementById('train-preload').addEventListener('change', async (e) => {
        const group = document.getElementById('preload-path-group');
        group.style.display = e.target.checked ? 'block' : 'none';
        
        if (e.target.checked) {
            await loadModelOptions();
        }
    });
}

async function saveTrainingConfig() {
    try {
        const settings = await apiCall('/settings');
        
        // Update settings from form
        settings.upscale_factor = parseInt(document.getElementById('train-upscale').value);
        settings.epochs_number = parseInt(document.getElementById('train-epochs').value);
        settings.batch_size = parseInt(document.getElementById('train-batch-size').value);
        settings.learning_rate = parseFloat(document.getElementById('train-lr').value);
        settings.optimizer = document.getElementById('train-optimizer').value;
        settings.tuning = document.getElementById('train-tuning').checked;
        settings.mixed_precision = document.getElementById('train-mixed-precision').checked;
        settings.scheduler = document.getElementById('train-scheduler').checked;
        settings.separable = document.getElementById('train-separable').checked;
        settings.optimized = document.getElementById('train-optimized').checked;
        settings.pruning = document.getElementById('train-pruning').checked;
        settings.prune_amount = parseFloat(document.getElementById('train-prune-amount').value);
        settings.preload = document.getElementById('train-preload').checked;
        
        if (settings.preload) {
            settings.preload_path = document.getElementById('train-preload-path').value;
        }
        
        // Device settings
        const device = document.querySelector('input[name="device"]:checked')?.value;
        if (device) {
            settings.cuda = device === 'cuda';
            settings.mps = device === 'mps';
        }
        
        await apiCall('/settings', 'POST', settings);
        showToast('Configuration saved!', 'success');
    } catch (error) {
        showToast('Failed to save configuration', 'error');
    }
}

async function loadModelOptions() {
    try {
        const models = await apiCall('/models');
        const select = document.getElementById('train-preload-path');
        select.innerHTML = '<option value="">Select a model...</option>';
        
        models.forEach(model => {
            model.checkpoints.forEach(ckp => {
                const option = document.createElement('option');
                option.value = ckp.path;
                option.textContent = `${model.name} - ${ckp.name}`;
                select.appendChild(option);
            });
        });
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

// ============================================
// Logs
// ============================================
function initLogsStream() {
    if (state.logsEventSource) {
        state.logsEventSource.close();
    }
    
    state.logsEventSource = new EventSource('/api/training/logs/stream');
    
    state.logsEventSource.onmessage = (event) => {
        const log = JSON.parse(event.data);
        appendLog(log);
    };
}

function appendLog(log) {
    const container = document.getElementById('logs-container');
    const placeholder = container.querySelector('.log-placeholder');
    if (placeholder) placeholder.remove();
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    const timestamp = new Date(log.timestamp).toLocaleTimeString();
    let messageClass = '';
    if (log.message.toLowerCase().includes('error')) messageClass = 'error';
    else if (log.message.toLowerCase().includes('warning')) messageClass = 'warning';
    else if (log.message.toLowerCase().includes('success') || log.message.toLowerCase().includes('saved')) messageClass = 'success';
    
    entry.innerHTML = `
        <span class="log-timestamp">${timestamp}</span>
        <span class="log-message ${messageClass}">${log.message}</span>
    `;
    
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

function initClearLogs() {
    document.getElementById('btn-clear-logs').addEventListener('click', () => {
        const container = document.getElementById('logs-container');
        container.innerHTML = '<div class="log-placeholder">No logs yet. Start training to see output.</div>';
    });
}

// ============================================
// Settings
// ============================================
async function loadSettings() {
    try {
        const settings = await apiCall('/settings');
        state.settings = settings;
        
        // Update YAML editor
        document.getElementById('yaml-editor').value = jsyaml.dump(settings);
        
        // Update visual form
        generateSettingsForm(settings);
        
        // Update training form
        updateTrainingForm(settings);
    } catch (error) {
        showToast('Failed to load settings', 'error');
    }
}

function generateSettingsForm(settings) {
    const container = document.getElementById('settings-form-container');
    let html = '';
    
    for (const [key, value] of Object.entries(settings)) {
        const inputId = `setting-${key}`;
        
        if (typeof value === 'boolean') {
            html += `
                <div class="form-group checkbox-group">
                    <label>
                        <input type="checkbox" id="${inputId}" ${value ? 'checked' : ''}>
                        <span>${formatKey(key)}</span>
                    </label>
                </div>
            `;
        } else if (typeof value === 'number') {
            html += `
                <div class="form-group">
                    <label for="${inputId}">${formatKey(key)}</label>
                    <input type="number" id="${inputId}" value="${value}" step="any">
                </div>
            `;
        } else if (typeof value === 'string') {
            html += `
                <div class="form-group">
                    <label for="${inputId}">${formatKey(key)}</label>
                    <input type="text" id="${inputId}" value="${value}">
                </div>
            `;
        }
    }
    
    container.innerHTML = html;
}

function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function updateTrainingForm(settings) {
    document.getElementById('train-upscale').value = settings.upscale_factor;
    document.getElementById('train-epochs').value = settings.epochs_number;
    document.getElementById('train-batch-size').value = settings.batch_size;
    document.getElementById('train-lr').value = settings.learning_rate;
    document.getElementById('train-optimizer').value = settings.optimizer;
    document.getElementById('train-tuning').checked = settings.tuning;
    document.getElementById('train-mixed-precision').checked = settings.mixed_precision;
    document.getElementById('train-scheduler').checked = settings.scheduler;
    document.getElementById('train-separable').checked = settings.separable;
    document.getElementById('train-optimized').checked = settings.optimized;
    document.getElementById('train-pruning').checked = settings.pruning;
    document.getElementById('train-prune-amount').value = settings.prune_amount;
    document.getElementById('prune-amount-value').textContent = settings.prune_amount;
    document.getElementById('train-preload').checked = settings.preload;
    
    // Device
    if (settings.cuda) {
        document.querySelector('input[name="device"][value="cuda"]').checked = true;
    } else if (settings.mps) {
        document.querySelector('input[name="device"][value="mps"]').checked = true;
    } else {
        document.querySelector('input[name="device"][value="cpu"]').checked = true;
    }
    
    // Show/hide conditional fields
    document.getElementById('prune-amount-group').style.display = 
        settings.pruning ? 'block' : 'none';
    document.getElementById('preload-path-group').style.display = 
        settings.preload ? 'block' : 'none';
}

function initSettingsControls() {
    document.getElementById('btn-reload-settings').addEventListener('click', loadSettings);
    
    document.getElementById('btn-save-yaml').addEventListener('click', async () => {
        try {
            const yamlText = document.getElementById('yaml-editor').value;
            const settings = jsyaml.load(yamlText);
            await apiCall('/settings', 'POST', settings);
            showToast('Settings saved successfully!', 'success');
            await apiCall('/settings/reset', 'POST');
        } catch (error) {
            showToast('Failed to save settings: ' + error.message, 'error');
        }
    });
}

// ============================================
// Models
// ============================================
async function loadModels() {
    try {
        const models = await apiCall('/models');
        state.models = models;
        renderModels(models);
    } catch (error) {
        showToast('Failed to load models', 'error');
    }
}

function renderModels(models) {
    const grid = document.getElementById('models-grid');
    
    if (models.length === 0) {
        grid.innerHTML = `
            <div class="card" style="grid-column: 1/-1; text-align: center; padding: 3rem;">
                <i class="fas fa-cube" style="font-size: 3rem; color: var(--text-muted); margin-bottom: 1rem;"></i>
                <p style="color: var(--text-secondary);">No trained models found. Start training to create models.</p>
            </div>
        `;
        return;
    }
    
    grid.innerHTML = models.map(model => {
        const tags = [];
        if (model.name.includes('mps')) tags.push('<span class="model-tag mps">MPS</span>');
        if (model.name.includes('cuda')) tags.push('<span class="model-tag cuda">CUDA</span>');
        if (model.name.includes('tuning')) tags.push('<span class="model-tag">Tuned</span>');
        if (model.name.includes('separable')) tags.push('<span class="model-tag">Separable</span>');
        
        return `
            <div class="model-card">
                <div class="model-card-header">
                    <h4>${model.name}</h4>
                    <div class="model-meta">
                        ${tags.join('')}
                    </div>
                </div>
                <div class="model-card-body">
                    <p style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 1rem;">
                        ${model.checkpoints.length} checkpoint(s)
                    </p>
                    <div class="checkpoint-list">
                        ${model.checkpoints.slice(0, 5).map(ckp => `
                            <div class="checkpoint-item">
                                <span class="checkpoint-name">${ckp.name}</span>
                                <div class="checkpoint-actions">
                                    <button class="btn btn-icon" onclick="loadModel('${ckp.path}')" title="Load Model">
                                        <i class="fas fa-upload"></i>
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                        ${model.checkpoints.length > 5 ? `
                            <p style="color: var(--text-muted); font-size: 0.75rem; text-align: center; margin-top: 0.5rem;">
                                +${model.checkpoints.length - 5} more checkpoints
                            </p>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

async function loadModel(path) {
    try {
        await apiCall('/model/load', 'POST', { model_path: path });
        showToast('Model path set! Ready for demo or continued training.', 'success');
    } catch (error) {
        showToast('Failed to set model path', 'error');
    }
}

// ============================================
// Results
// ============================================
async function loadResults() {
    try {
        const results = await apiCall('/results');
        state.results = results;
        renderResultSets(results);
    } catch (error) {
        showToast('Failed to load results', 'error');
    }
}

function renderResultSets(results) {
    const list = document.getElementById('result-sets');
    
    if (results.length === 0) {
        list.innerHTML = '<li class="result-set-item" style="color: var(--text-muted);">No results yet</li>';
        return;
    }
    
    list.innerHTML = results.map(result => `
        <li class="result-set-item" data-path="${result.path}" onclick="selectResultSet('${result.path}')">
            ${result.name}
            <span style="display: block; font-size: 0.75rem; color: var(--text-muted);">
                ${result.images.length} images
            </span>
        </li>
    `).join('');
}

function selectResultSet(path) {
    // Update active state
    document.querySelectorAll('.result-set-item').forEach(item => {
        item.classList.toggle('active', item.dataset.path === path);
    });
    
    // Find result set
    const resultSet = state.results.find(r => r.path === path);
    if (!resultSet) return;
    
    state.currentResultSet = resultSet;
    state.lightboxImages = resultSet.images;
    
    renderGallery(resultSet.images);
}

function renderGallery(images) {
    const gallery = document.getElementById('image-gallery');
    
    if (images.length === 0) {
        gallery.innerHTML = `
            <div class="gallery-placeholder">
                <i class="fas fa-images"></i>
                <p>No images in this result set</p>
            </div>
        `;
        return;
    }
    
    gallery.innerHTML = images.map((img, index) => `
        <div class="gallery-item" onclick="openLightbox(${index})">
            <img src="/api/results/${img.path}" alt="${img.name}" loading="lazy">
            <span class="gallery-item-name">${img.name}</span>
        </div>
    `).join('');
}

function openLightbox(index) {
    state.lightboxIndex = index;
    const lightbox = document.getElementById('lightbox');
    const image = document.getElementById('lightbox-image');
    const info = document.getElementById('lightbox-info');
    
    const img = state.lightboxImages[index];
    image.src = `/api/results/${img.path}`;
    info.textContent = `${img.name} (${index + 1}/${state.lightboxImages.length})`;
    
    lightbox.classList.add('active');
}

function closeLightbox() {
    document.getElementById('lightbox').classList.remove('active');
}

function prevImage() {
    state.lightboxIndex = (state.lightboxIndex - 1 + state.lightboxImages.length) % state.lightboxImages.length;
    updateLightboxImage();
}

function nextImage() {
    state.lightboxIndex = (state.lightboxIndex + 1) % state.lightboxImages.length;
    updateLightboxImage();
}

function updateLightboxImage() {
    const image = document.getElementById('lightbox-image');
    const info = document.getElementById('lightbox-info');
    const img = state.lightboxImages[state.lightboxIndex];
    
    image.src = `/api/results/${img.path}`;
    info.textContent = `${img.name} (${state.lightboxIndex + 1}/${state.lightboxImages.length})`;
}

function initLightbox() {
    document.querySelector('.lightbox-close').addEventListener('click', closeLightbox);
    document.querySelector('.lightbox-prev').addEventListener('click', prevImage);
    document.querySelector('.lightbox-next').addEventListener('click', nextImage);
    
    document.getElementById('lightbox').addEventListener('click', (e) => {
        if (e.target.id === 'lightbox') closeLightbox();
    });
    
    document.addEventListener('keydown', (e) => {
        const lightbox = document.getElementById('lightbox');
        if (!lightbox.classList.contains('active')) return;
        
        if (e.key === 'Escape') closeLightbox();
        if (e.key === 'ArrowLeft') prevImage();
        if (e.key === 'ArrowRight') nextImage();
    });
}

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initStatsStream();
    initLogsStream();
    initTrainingControls();
    initClearLogs();
    initSettingsControls();
    initLightbox();
    
    // Initial data load
    loadSettings();
});

// Make functions available globally for onclick handlers
window.loadModel = loadModel;
window.selectResultSet = selectResultSet;
window.openLightbox = openLightbox;
