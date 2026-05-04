// ============================================
// DOM Elements
// ============================================

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const clearBtn = document.getElementById('clearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const alertBanner = document.getElementById('alertBanner');
const alertMessage = document.getElementById('alertMessage');
const helmetCount = document.getElementById('helmetCount');
const headCount = document.getElementById('headCount');
const personCount = document.getElementById('personCount');
const resultImage = document.getElementById('resultImage');
const detectionTableBody = document.getElementById('detectionTableBody');
const noDetections = document.getElementById('noDetections');
const resetBtn = document.getElementById('resetBtn');
const toastContainer = document.getElementById('toastContainer');

// State
let selectedFile = null;

// ============================================
// File Upload Handlers
// ============================================

// Click to browse
uploadZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// ============================================
// File Selection & Validation
// ============================================

function handleFileSelect(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        showToast('Please select a valid image file (JPG or PNG)', 'error');
        return;
    }

    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('File size exceeds 16MB limit', 'error');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadZone.classList.add('hidden');
        previewSection.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

// ============================================
// Clear & Reset Handlers
// ============================================

clearBtn.addEventListener('click', () => {
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadZone.classList.remove('hidden');
    previewSection.classList.add('hidden');
}

resetBtn.addEventListener('click', () => {
    resetUpload();
    resultsSection.classList.add('hidden');
    uploadZone.scrollIntoView({ behavior: 'smooth', block: 'center' });
});

// ============================================
// Analyze Image
// ============================================

analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) {
        showToast('Please select an image first', 'error');
        return;
    }

    performAnalysis();
});

async function performAnalysis() {
    // Show loading
    loadingSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Analysis failed');
        }

        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        showToast(error.message || 'Failed to analyze image. Please try again.', 'error');
    } finally {
        loadingSection.classList.add('hidden');
    }
}

// ============================================
// Display Results
// ============================================

function displayResults(data) {
    const { detections, counts, violation, image_url } = data;

    // Update alert banner
    alertBanner.className = 'alert-banner';
    if (violation) {
        alertBanner.classList.add('violation');
        alertMessage.textContent = '⚠️ Safety violation detected! Some workers are not wearing helmets.';
    } else {
        alertBanner.classList.add('safe');
        alertMessage.textContent = '✓ All safety compliant! All workers are wearing helmets.';
    }

    // Update stats cards with animation
    animateCount(helmetCount, counts.Helmet || 0);
    animateCount(headCount, counts.Head || 0);
    animateCount(personCount, counts.Person || 0);

    // Set annotated image
    resultImage.src = image_url;

    // Populate detection table
    populateDetectionTable(detections);

    // Show results section
    resultsSection.classList.remove('hidden');

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function animateCount(element, target) {
    const duration = 500;
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // Ease out cubic
        const current = Math.round(start + (target - start) * eased);
        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function populateDetectionTable(detections) {
    detectionTableBody.innerHTML = '';

    if (!detections || detections.length === 0) {
        noDetections.classList.remove('hidden');
        return;
    }

    noDetections.classList.add('hidden');

    detections.forEach((detection, index) => {
        const row = document.createElement('tr');

        const classBadge = getClassBadge(detection.class_name);
        const confidencePercent = Math.round(detection.confidence * 100);
        const confidenceBar = getConfidenceBar(detection.confidence);
        const bbox = formatBBox(detection.bbox);

        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${classBadge}</td>
            <td>
                <div class="confidence-bar">
                    ${confidenceBar}
                    <span>${confidencePercent}%</span>
                </div>
            </td>
            <td><code style="font-size: 0.8em; color: var(--text-secondary);">${bbox}</code></td>
        `;

        detectionTableBody.appendChild(row);
    });
}

function getClassBadge(className) {
    const classMap = {
        'Helmet': 'helmet',
        'Head': 'head',
        'Person': 'person'
    };

    const badgeClass = classMap[className] || 'person';
    return `<span class="class-badge class-badge--${badgeClass}">${className}</span>`;
}

function getConfidenceBar(confidence) {
    const percent = Math.round(confidence * 100);
    return `
        <div class="confidence-bar-track">
            <div class="confidence-bar-fill" style="width: ${percent}%"></div>
        </div>
    `;
}

function formatBBox(bbox) {
    if (!bbox || bbox.length !== 4) return 'N/A';
    const [x1, y1, x2, y2] = bbox;
    return `[${Math.round(x1)}, ${Math.round(y1)}, ${Math.round(x2)}, ${Math.round(y2)}]`;
}

// ============================================
// Toast Notifications
// ============================================

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;

    const icon = type === 'success'
        ? '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        : '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><path d="M12 8V12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="16" r="1" fill="currentColor"/></svg>';

    toast.innerHTML = `
        ${icon}
        <span class="toast-message">${message}</span>
    `;

    toastContainer.appendChild(toast);

    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.classList.add('removing');
        toast.addEventListener('animationend', () => {
            toast.remove();
        });
    }, 4000);
}

// ============================================
// Initialize
// ============================================

// Prevent default drag behaviors on document
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    document.body.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    });
});

console.log('Hard Hat Detector frontend initialized');
