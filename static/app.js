document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const tabSingle = document.getElementById('tab-single');
    const tabFolder = document.getElementById('tab-folder');
    const dropZone = document.getElementById('drop-zone');
    const fileInputSingle = document.getElementById('file-input-single');
    const fileInputFolder = document.getElementById('file-input-folder');
    const dropMainText = document.getElementById('drop-main-text');
    const dropSubText = document.getElementById('drop-sub-text');
    const selectedBadge = document.getElementById('selected-badge');
    const selectedCount = document.getElementById('selected-count');
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');
    const iouSlider = document.getElementById('iou-slider');
    const iouVal = document.getElementById('iou-val');
    const detectBtn = document.getElementById('detect-btn');
    const clearBtn = document.getElementById('clear-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');

    const statsBar = document.getElementById('stats-bar');
    const statTotal = document.getElementById('stat-total');
    const statWearing = document.getElementById('stat-wearing');
    const statIncorrect = document.getElementById('stat-incorrect');
    const statNoMask = document.getElementById('stat-nomask');
    const statLatency = document.getElementById('stat-latency');

    const emptyState = document.getElementById('empty-state');
    const resultsContainer = document.getElementById('results-container');
    const terminalLog = document.getElementById('terminal-log');

    const imageModal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    const modalCaption = document.getElementById('modal-caption');
    const modalClose = document.getElementById('modal-close');

    let currentMode = 'single'; // 'single' or 'folder'
    let selectedFiles = [];

    // Slider Listeners
    confSlider.addEventListener('input', (e) => {
        confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    iouSlider.addEventListener('input', (e) => {
        iouVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Tab Switching
    tabSingle.addEventListener('click', () => {
        currentMode = 'single';
        tabSingle.classList.add('active');
        tabFolder.classList.remove('active');
        dropMainText.textContent = 'Drag & drop your image here';
        dropSubText.textContent = 'or click to browse files (JPG, PNG, WEBP)';
        resetFiles();
    });

    tabFolder.addEventListener('click', () => {
        currentMode = 'folder';
        tabFolder.classList.add('active');
        tabSingle.classList.remove('active');
        dropMainText.textContent = 'Drag & drop image folder here';
        dropSubText.textContent = 'or click to select an entire folder of photos';
        resetFiles();
    });

    // Dropzone Click
    dropZone.addEventListener('click', () => {
        if (currentMode === 'single') {
            fileInputSingle.click();
        } else {
            fileInputFolder.click();
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = Array.from(dt.files).filter(f => f.type.startsWith('image/') || f.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i));
        handleFiles(files);
    });

    fileInputSingle.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleFiles(files);
    });

    fileInputFolder.addEventListener('change', (e) => {
        const files = Array.from(e.target.files).filter(f => f.type.startsWith('image/') || f.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i));
        handleFiles(files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        selectedFiles = files;
        selectedBadge.style.display = 'inline-block';
        selectedCount.textContent = files.length === 1 
            ? files[0].name 
            : `${files.length} images ready for detection`;
        detectBtn.disabled = false;
        logToTerminal(`Loaded ${files.length} image(s). Ready to run inference.`);
    }

    function resetFiles() {
        selectedFiles = [];
        fileInputSingle.value = '';
        fileInputFolder.value = '';
        selectedBadge.style.display = 'none';
        detectBtn.disabled = true;
    }

    clearBtn.addEventListener('click', () => {
        resetFiles();
        resultsContainer.innerHTML = '';
        resultsContainer.style.display = 'none';
        emptyState.style.display = 'flex';
        statsBar.style.display = 'none';
        logToTerminal('Cleared all inputs and results.');
    });

    // Detect Action
    detectBtn.addEventListener('click', async () => {
        if (selectedFiles.length === 0) return;

        detectBtn.disabled = true;
        btnSpinner.style.display = 'inline-block';
        btnText.textContent = 'Processing...';
        logToTerminal(`Starting inference on ${selectedFiles.length} image(s) [conf=${confSlider.value}, iou=${iouSlider.value}]...`);

        const formData = new FormData();
        selectedFiles.forEach((file) => {
            formData.append('files', file);
        });
        formData.append('conf', confSlider.value);
        formData.append('iou', iouSlider.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Server processing error');
            }

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            logToTerminal(`[ERROR] ${error.message}`);
            alert(`Error running detection: ${error.message}`);
        } finally {
            detectBtn.disabled = false;
            btnSpinner.style.display = 'none';
            btnText.textContent = 'Detect Face Masks';
        }
    });

    function renderResults(data) {
        emptyState.style.display = 'none';
        resultsContainer.innerHTML = '';
        resultsContainer.style.display = 'flex';

        // Update Aggregate Stats
        statsBar.style.display = 'grid';
        statTotal.textContent = data.aggregate.total_faces;
        statWearing.textContent = data.aggregate.wearing_mask;
        statIncorrect.textContent = data.aggregate.incorrect_mask;
        statNoMask.textContent = data.aggregate.no_mask;
        statLatency.textContent = `${data.aggregate.avg_latency_ms.toFixed(1)} ms`;

        let logOutput = `============================================================\n`;
        logOutput += `INFERENCE BATCH COMPLETE (${data.results.length} images processed)\n`;
        logOutput += `Total Faces Detected : ${data.aggregate.total_faces}\n`;
        logOutput += `  - Wearing Mask     : ${data.aggregate.wearing_mask}\n`;
        logOutput += `  - Incorrect Mask   : ${data.aggregate.incorrect_mask}\n`;
        logOutput += `  - No Mask          : ${data.aggregate.no_mask}\n`;
        logOutput += `Average Latency      : ${data.aggregate.avg_latency_ms.toFixed(1)} ms/image\n`;
        logOutput += `============================================================\n\n`;

        data.results.forEach((res, index) => {
            logOutput += `[${index + 1}/${data.results.length}] ${res.filename} -> ${res.total_faces} face(s) | Wearing: ${res.counts['Wearing Mask']}, Incorrect: ${res.counts['Wore Mask Incorrectly']}, No Mask: ${res.counts['No Mask']} (${res.latency_ms}ms)\n`;

            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <div class="result-image-box" onclick="openZoomModal('${res.image_base64}', '${res.filename}')">
                    <img src="${res.image_base64}" alt="Result ${res.filename}">
                    <span class="zoom-hint">🔍 Click to zoom</span>
                </div>
                <div class="result-details">
                    <div>
                        <div class="detail-header">
                            <h4>${res.filename}</h4>
                            <div class="detail-meta">Detected in ${res.latency_ms} ms • ${res.total_faces} Face(s)</div>
                        </div>
                        <div class="detection-chips">
                            <div class="chip chip-green">
                                <span>Wearing Mask</span>
                                <strong>${res.counts['Wearing Mask']}</strong>
                            </div>
                            <div class="chip chip-yellow">
                                <span>Wore Mask Incorrectly</span>
                                <strong>${res.counts['Wore Mask Incorrectly']}</strong>
                            </div>
                            <div class="chip chip-red">
                                <span>No Mask</span>
                                <strong>${res.counts['No Mask']}</strong>
                            </div>
                        </div>
                    </div>
                    <div class="detail-footer">
                        <a href="${res.image_base64}" download="annotated_${res.filename}" class="btn-download">💾 Download Image</a>
                    </div>
                </div>
            `;
            resultsContainer.appendChild(card);
        });

        terminalLog.textContent = logOutput;
        terminalLog.scrollTop = terminalLog.scrollHeight;
    }

    function logToTerminal(message) {
        const time = new Date().toLocaleTimeString();
        terminalLog.textContent += `\n[${time}] ${message}`;
        terminalLog.scrollTop = terminalLog.scrollHeight;
    }

    // Modal Handling
    window.openZoomModal = function(src, caption) {
        modalImg.src = src;
        modalCaption.textContent = caption;
        imageModal.style.display = 'block';
    };

    modalClose.addEventListener('click', () => {
        imageModal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === imageModal) {
            imageModal.style.display = 'none';
        }
    });
});
