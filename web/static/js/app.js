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
function getTuningFormValues() {
    const tuningCheckbox = document.getElementById('train-enable-tuning');
    const trialsInput = document.getElementById('train-trials');

    const trialsValue = trialsInput ? parseInt(trialsInput.value, 10) : NaN;

    return {
        tuning: tuningCheckbox ? tuningCheckbox.checked : undefined,
        trials: Number.isFinite(trialsValue) ? trialsValue : undefined
    };
}

function initTrainingControls() {
    // Start Training button (Dashboard)
    document.getElementById('btn-start-training').addEventListener('click', async () => {
        try {
            const { tuning, trials } = getTuningFormValues();
            const payload = { mode: 'train' };
            if (tuning !== undefined) payload.tuning = tuning;
            if (trials !== undefined) payload.trials = trials;

            await apiCall('/training/start', 'POST', payload);
            showToast('Training started!', 'success');
        } catch (error) {
            showToast(error.message, 'error');
        }
    });

    // Start Training button (Training tab)
    document.getElementById('btn-start-training-2').addEventListener('click', async () => {
        await saveTrainingConfig();
        try {
            const { tuning, trials } = getTuningFormValues();
            const payload = { mode: 'train' };
            if (tuning !== undefined) payload.tuning = tuning;
            if (trials !== undefined) payload.trials = trials;

            await apiCall('/training/start', 'POST', payload);
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

    // Dropout rate slider
    document.getElementById('train-dropout-rate').addEventListener('input', (e) => {
        document.getElementById('dropout-rate-value').textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Object-Aware ESPCN toggle
    document.getElementById('train-optimized').addEventListener('change', (e) => {
        document.getElementById('num-classes-group').style.display =
            e.target.checked ? 'block' : 'none';
    });

    // Early Stopping toggle
    document.getElementById('train-early-stopping').addEventListener('change', (e) => {
        const group = document.getElementById('early-stopping-group');
        const targetGroup = document.getElementById('target-psnr-group');
        const stuckGroup = document.getElementById('stuck-level-group');
        group.style.display = e.target.checked ? 'block' : 'none';
        targetGroup.style.display = e.target.checked ? 'block' : 'none';
        stuckGroup.style.display = e.target.checked ? 'block' : 'none';
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
        settings.epoch = parseInt(document.getElementById('train-epochs').value); // Keep both for compatibility
        settings.batch_size = parseInt(document.getElementById('train-batch-size').value);
        settings.learning_rate = parseFloat(document.getElementById('train-lr').value);
        settings.momentum = parseFloat(document.getElementById('train-momentum').value);
        settings.weight_decay = parseFloat(document.getElementById('train-weight-decay').value);
        settings.optimizer = document.getElementById('train-optimizer').value;
        settings.checkpoint_frequency = parseInt(document.getElementById('train-checkpoint-freq').value);
        settings.mixed_precision = document.getElementById('train-mixed-precision').checked;
        settings.scheduler = document.getElementById('train-scheduler').checked;
        settings.channels_last = document.getElementById('train-channels-last').checked;
        settings.compile_model = document.getElementById('train-compile-model').checked;
        settings.gradient_accumulation_steps = parseInt(document.getElementById('train-gradient-accum').value);
        settings.tuning = document.getElementById('train-enable-tuning').checked;
        const trialsValue = parseInt(document.getElementById('train-trials').value);
        if (!Number.isNaN(trialsValue)) {
            settings.trials = trialsValue;
        }

        // Model architecture
        settings.separable = document.getElementById('train-separable').checked;
        settings.use_bn = document.getElementById('train-use-bn').checked;
        settings.dropout_rate = parseFloat(document.getElementById('train-dropout-rate').value);
        settings.optimized = document.getElementById('train-optimized').checked;
        settings.num_classes = parseInt(document.getElementById('train-num-classes').value);

        // Advanced optimizations
        settings.cache_dataset = document.getElementById('train-cache-dataset').checked;
        settings.use_fused_optimizer = document.getElementById('train-fused-optimizer').checked;
        settings.persistent_workers = document.getElementById('train-persistent-workers').checked;

        // Early stopping
        settings.early_stopping = document.getElementById('train-early-stopping').checked;
        settings.early_stopping_patience = parseInt(document.getElementById('train-es-patience').value);
        settings.target_min_psnr = parseFloat(document.getElementById('train-target-psnr').value);
        settings.stuck_level = parseInt(document.getElementById('train-stuck-level').value);

        // Pruning
        settings.pruning = document.getElementById('train-pruning').checked;
        settings.prune_amount = parseFloat(document.getElementById('train-prune-amount').value);

        // Preload
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
    document.getElementById('train-epochs').value = settings.epochs_number || settings.epoch;
    document.getElementById('train-batch-size').value = settings.batch_size;
    document.getElementById('train-lr').value = settings.learning_rate;
    document.getElementById('train-momentum').value = settings.momentum || 0.9;
    document.getElementById('train-weight-decay').value = settings.weight_decay || 0.0;
    document.getElementById('train-optimizer').value = settings.optimizer;
    document.getElementById('train-checkpoint-freq').value = settings.checkpoint_frequency || 100;
    document.getElementById('train-mixed-precision').checked = settings.mixed_precision || false;
    document.getElementById('train-scheduler').checked = settings.scheduler !== false;
    document.getElementById('train-channels-last').checked = settings.channels_last || false;
    document.getElementById('train-compile-model').checked = settings.compile_model || false;
    document.getElementById('train-gradient-accum').value = settings.gradient_accumulation_steps || 1;
    document.getElementById('train-enable-tuning').checked = settings.tuning || false;
    document.getElementById('train-trials').value = settings.trials || 50;

    // Model architecture
    document.getElementById('train-separable').checked = settings.separable !== false;
    document.getElementById('train-use-bn').checked = settings.use_bn || false;
    document.getElementById('train-dropout-rate').value = settings.dropout_rate || 0;
    document.getElementById('dropout-rate-value').textContent = (settings.dropout_rate || 0).toFixed(2);
    document.getElementById('train-optimized').checked = settings.optimized || false;
    document.getElementById('train-num-classes').value = settings.num_classes || 5;

    // Show/hide optimized options
    document.getElementById('num-classes-group').style.display =
        settings.optimized ? 'block' : 'none';

    // Advanced optimizations
    document.getElementById('train-cache-dataset').checked = settings.cache_dataset || false;
    document.getElementById('train-fused-optimizer').checked = settings.use_fused_optimizer || false;
    document.getElementById('train-persistent-workers').checked = settings.persistent_workers || false;

    // Early stopping
    document.getElementById('train-early-stopping').checked = settings.early_stopping !== false;
    document.getElementById('train-es-patience').value = settings.early_stopping_patience || 50;
    document.getElementById('train-target-psnr').value = settings.target_min_psnr || 26;
    document.getElementById('train-stuck-level').value = settings.stuck_level || 30;

    // Pruning
    document.getElementById('train-pruning').checked = settings.pruning || false;
    document.getElementById('train-prune-amount').value = settings.prune_amount || 0.1;
    document.getElementById('prune-amount-value').textContent = (settings.prune_amount || 0.1).toFixed(2);

    // Show/hide conditional fields
    document.getElementById('prune-amount-group').style.display =
        settings.pruning ? 'block' : 'none';
    document.getElementById('preload-path-group').style.display =
        settings.preload ? 'block' : 'none';

    // Device
    if (settings.cuda) {
        const cudaRadio = document.querySelector('input[name="device"][value="cuda"]');
        if (cudaRadio) cudaRadio.checked = true;
    } else if (settings.mps) {
        const mpsRadio = document.querySelector('input[name="device"][value="mps"]');
        if (mpsRadio) mpsRadio.checked = true;
    } else {
        const cpuRadio = document.querySelector('input[name="device"][value="cpu"]');
        if (cpuRadio) cpuRadio.checked = true;
    }
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
                        <button class="btn btn-icon" title="Delete model" onclick="deleteModel('${model.name}')">
                            <i class="fas fa-trash"></i>
                        </button>
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

async function deleteModel(name) {
    if (!confirm(`Delete model "${name}"? This will remove all checkpoints and logs.`)) return;
    try {
        await apiCall('/models/delete', 'POST', { name });
        showToast('Model deleted', 'success');
        loadModels();
    } catch (error) {
        showToast(error.message, 'error');
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
            <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
                <div>
                    ${result.name}
                    <span style="display: block; font-size: 0.75rem; color: var(--text-muted);">
                        ${result.images.length} images
                    </span>
                </div>
                <button class="btn btn-icon" title="Delete results" onclick="deleteResultSet(event, '${result.name}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
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

async function deleteResultSet(event, name) {
    event.stopPropagation();
    if (!confirm(`Delete result set "${name}"?`)) return;
    try {
        await apiCall('/results/delete', 'POST', { name });
        showToast('Result set deleted', 'success');
        await loadResults();
        state.currentResultSet = null;
        const gallery = document.getElementById('image-gallery');
        gallery.innerHTML = `
            <div class="gallery-placeholder">
                <i class="fas fa-images"></i>
                <p>Select a result set to view images</p>
            </div>
        `;
    } catch (error) {
        showToast(error.message, 'error');
    }
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
// Auto-Config Functions
// ============================================
async function detectHardware() {
    const loading = document.getElementById('hardware-loading');
    const specs = document.getElementById('hardware-specs');
    const error = document.getElementById('hardware-error');
    const recommendationsGrid = document.getElementById('recommendations-grid');

    // Show loading state
    specs.style.display = 'none';
    error.style.display = 'none';
    recommendationsGrid.style.display = 'none';
    loading.style.display = 'block';

    try {
        const upscaleFactor = parseInt(document.getElementById('autoconfig-upscale-factor').value);
        const result = await apiCall(`/autoconfig/detect?upscale_factor=${upscaleFactor}`);

        if (result.success) {
            const hw = result.hardware;
            const recommendations = result.recommendations;

            // Update hardware specs
            document.getElementById('spec-platform').textContent = `${hw.platform}`;
            document.getElementById('spec-platform-release').textContent = `${hw.platform_release}`;
            document.getElementById('spec-cpu-physical').textContent = `${hw.cpu_cores.physical}`;
            document.getElementById('spec-cpu-logical').textContent = `${hw.cpu_cores.logical}`;
            document.getElementById('spec-ram-total').textContent = `${hw.ram_gb.total}`;
            document.getElementById('spec-ram-available').textContent = `${hw.ram_gb.available}`;

            const gpuDetails = document.getElementById('gpu-details');
            if (hw.gpu.has_cuda) {
                gpuDetails.innerHTML = `<strong>${hw.gpu.cuda_device_name}</strong><br>VRAM: ${hw.gpu.cuda_memory_gb} GB`;
            } else if (hw.gpu.has_mps) {
                gpuDetails.innerHTML = `<strong>Apple Neural Engine (MPS)</strong>`;
            } else {
                gpuDetails.innerHTML = `<em>No dedicated GPU detected</em>`;
            }

            // Update device badge
            const deviceBadge = document.getElementById('spec-device');
            deviceBadge.textContent = hw.gpu.recommended_device.toUpperCase();
            deviceBadge.className = `badge ${hw.gpu.recommended_device}`;

            // Update tier
            const tierName = document.getElementById('tier-name');
            const tierDisplay = document.getElementById('tier-display');
            const tierBadge = document.getElementById('tier-badge');

            tierName.textContent = recommendations.tier;
            tierBadge.className = `tier-badge ${recommendations.tier.toLowerCase()}`;

            const tierDescriptions = {
                'ULTRA': 'High-end GPU with ample RAM. Maximum performance, all optimizations enabled.',
                'HIGH': 'Good GPU and CPU/RAM balance. Fast training with modern optimizations.',
                'MEDIUM': 'Moderate resources. Balanced settings with smart memory management.',
                'LOW': 'Limited resources. Conservative settings with gradient accumulation.'
            };

            document.getElementById('tier-description').textContent = tierDescriptions[recommendations.tier] || '';
            tierDisplay.style.display = 'block';

            // Update recommendations
            updateRecommendations(recommendations);

            // Hide error and loading
            loading.style.display = 'none';
            specs.style.display = 'block';
            recommendationsGrid.style.display = 'grid';
            error.style.display = 'none';

            showToast('Hardware detected successfully!', 'success');
        } else {
            throw new Error(result.error || 'Detection failed');
        }
    } catch (err) {
        loading.style.display = 'none';
        error.style.display = 'block';
        document.getElementById('hardware-error-message').textContent = err.message;
        showToast(`Error: ${err.message}`, 'error');
    }
}

function updateRecommendations(rec) {
    const recommendations = rec;

    // Device settings
    document.getElementById('rec-device').textContent = recommendations.device_settings.cuda ? 'CUDA' :
        recommendations.device_settings.mps ? 'MPS' : 'CPU';
    document.getElementById('rec-mixed-precision').textContent = recommendations.device_settings.mixed_precision ? 'Yes' : 'No';
    document.getElementById('rec-channels-last').textContent = recommendations.device_settings.channels_last ? 'Yes' : 'No';

    // Performance settings
    document.getElementById('rec-batch-size').textContent = recommendations.training_settings.batch_size;
    document.getElementById('rec-test-batch-size').textContent = recommendations.training_settings.test_batch_size;
    document.getElementById('rec-threads').textContent = recommendations.training_settings.threads;
    document.getElementById('rec-cache-dataset').textContent = recommendations.device_settings.cache_dataset ? 'Yes' : 'No';

    // Advanced optimizations
    document.getElementById('rec-compile').textContent = recommendations.device_settings.compile_model ?
        (recommendations.device_settings.compile_mode || 'default') : 'No';
    document.getElementById('rec-fused-opt').textContent = recommendations.device_settings.use_fused_optimizer ? 'Yes' : 'No';
    document.getElementById('rec-persistent').textContent = recommendations.device_settings.persistent_workers ? 'Yes' : 'No';

    // Store for later use
    state.currentRecommendations = recommendations.complete_config;
}

async function compareSettings() {
    const table = document.getElementById('comparison-table');
    const tbody = document.getElementById('comparison-tbody');
    const placeholder = document.querySelector('.comparison-placeholder');

    try {
        const result = await apiCall('/autoconfig/compare');

        if (result.success) {
            tbody.innerHTML = '';

            const comparison = result.comparison;
            let matchCount = 0;
            let totalCount = 0;

            for (const [category, items] of Object.entries(comparison)) {
                for (const [key, comparison_data] of Object.entries(items)) {
                    totalCount++;
                    const current = comparison_data.current;
                    const recommended = comparison_data.recommended;
                    const matches = comparison_data.matches;

                    if (matches) matchCount++;

                    const row = document.createElement('tr');
                    row.className = matches ? 'match' : 'mismatch';

                    const status = matches ?
                        '<span class="badge success"><i class="fas fa-check"></i> Match</span>' :
                        '<span class="badge warning"><i class="fas fa-exclamation"></i> Different</span>';

                    row.innerHTML = `
                        <td><strong>${key}</strong></td>
                        <td>${JSON.stringify(current)}</td>
                        <td>${JSON.stringify(recommended)}</td>
                        <td>${status}</td>
                    `;

                    tbody.appendChild(row);
                }
            }

            // Show summary
            showToast(`${matchCount}/${totalCount} settings match recommendations`, 'info');

            placeholder.style.display = 'none';
            table.style.display = 'block';
        } else {
            throw new Error(result.error || 'Comparison failed');
        }
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
}

async function applyAutoconfig() {
    const upscaleFactor = parseInt(document.getElementById('autoconfig-upscale-factor').value);
    const createBackup = document.getElementById('autoconfig-backup').checked;

    if (!confirm('Apply auto-configuration to settings.yaml? Your custom settings will be preserved.')) {
        return;
    }

    try {
        const result = await apiCall('/autoconfig/apply', 'POST', {
            upscale_factor: upscaleFactor,
            backup: createBackup
        });

        if (result.success) {
            showToast(`✓ Auto-configuration applied! (${result.tier} tier)`, 'success');
            // Reload settings
            await loadSettings();
        } else {
            throw new Error(result.error || 'Apply failed');
        }
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
}

async function downloadAutoconfig() {
    try {
        const upscaleFactor = parseInt(document.getElementById('autoconfig-upscale-factor').value);
        const result = await apiCall(`/autoconfig/detect?upscale_factor=${upscaleFactor}`);

        if (result.success) {
            const config = result.recommendations.complete_config;
            const yaml = JSON.stringify(config, null, 2); // Simple JSON representation

            // Create blob and download
            const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `settings_autoconfig_${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            showToast('Configuration downloaded!', 'success');
        } else {
            throw new Error(result.error || 'Download failed');
        }
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
}

function initAutoConfig() {
    const detectBtn = document.getElementById('btn-detect-hardware');
    const compareBtn = document.getElementById('btn-compare-settings');
    const applyBtn = document.getElementById('btn-apply-autoconfig');
    const downloadBtn = document.getElementById('btn-download-autoconfig');

    if (detectBtn) detectBtn.addEventListener('click', detectHardware);
    if (compareBtn) compareBtn.addEventListener('click', compareSettings);
    if (applyBtn) applyBtn.addEventListener('click', applyAutoconfig);
    if (downloadBtn) downloadBtn.addEventListener('click', downloadAutoconfig);

    // Auto-run detection when tab is opened
    const autoConfigTab = document.getElementById('autoconfig-tab');
    if (autoConfigTab) {
        // Use a simple flag to detect when tab is shown
        const observer = new MutationObserver(() => {
            if (autoConfigTab.classList.contains('active') && !state.hardwareDetected) {
                detectHardware();
                state.hardwareDetected = true;
            }
        });

        observer.observe(autoConfigTab, { attributes: true, attributeFilter: ['class'] });
    }
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
    initAutoConfig();
    initLightbox();

    // Initial data load
    loadSettings();
});

// Make functions available globally for onclick handlers
window.loadModel = loadModel;
window.selectResultSet = selectResultSet;
window.openLightbox = openLightbox;
