/**
 * Face Data Collector - Main JavaScript
 * Logic điều khiển camera và thu thập ảnh
 */

// ============ Constants ============
const MAX_CAPTURES = 60;
const PREVIEW_MAX = 5;

// ============ DOM Elements ============
const video = document.getElementById('collectVideo');
const canvas = document.getElementById('collectCanvas');
const empIdInput = document.getElementById('collectEmpId');
const nameInput = document.getElementById('collectName');
const toggleCameraBtn = document.getElementById('collectToggleCamera');
const switchCameraBtn = document.getElementById('collectSwitchCamera');
const captureBtn = document.getElementById('collectCaptureBtn');
const progressFill = document.getElementById('collectProgressFill');
const progressText = document.getElementById('collectProgressText');
const previewContainer = document.getElementById('collectPreviewContainer');
const cameraOverlay = document.getElementById('collectCameraOverlay');

// ============ State ============
let stream = null;
let isCameraOn = false;
let facingMode = 'user';
let captureCount = 0;
let isCapturing = false;

// ============ Toast Notification ============
function showToast(message, type = 'info') {
    const container = document.getElementById('collectToastContainer');
    const toast = document.createElement('div');
    toast.className = `collect-toast ${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                 type === 'error' ? 'fa-exclamation-circle' :
                 type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle';
    
    toast.innerHTML = `<i class="fas ${icon}"></i> ${message}`;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'collectSlideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============ Validation ============
function validateEmpId(empId) {
    const pattern = /^NV\d{2}$/i;
    return pattern.test(empId);
}

function formatEmpId(empId) {
    return empId.toUpperCase();
}

function validateName(name) {
    const pattern = /^[a-zA-Z_]+$/;
    return pattern.test(name);
}

function validateForm() {
    let empId = empIdInput.value.trim();
    const name = nameInput.value.trim();
    
    let isValid = true;
    
    const empIdError = document.getElementById('collectEmpIdError');
    if (!empId) {
        empIdInput.classList.remove('success', 'error');
        empIdError.classList.remove('show');
    } else if (!validateEmpId(empId)) {
        empIdInput.classList.remove('success');
        empIdInput.classList.add('error');
        empIdError.textContent = 'Mã NV phải có định dạng: NV + 2 chữ số (VD: NV01, NV02)';
        empIdError.classList.add('show');
        isValid = false;
    } else {
        empIdInput.value = formatEmpId(empId);
        empIdInput.classList.remove('error');
        empIdInput.classList.add('success');
        empIdError.classList.remove('show');
        updateCameraInfo();
    }
    
    const nameError = document.getElementById('collectNameError');
    if (!name) {
        nameInput.classList.remove('success', 'error');
        nameError.classList.remove('show');
    } else if (!validateName(name)) {
        nameInput.classList.remove('success');
        nameInput.classList.add('error');
        nameError.textContent = 'Họ tên chỉ chấp nhận chữ cái không dấu và gạch dưới';
        nameError.classList.add('show');
        isValid = false;
    } else {
        nameInput.classList.remove('error');
        nameInput.classList.add('success');
        nameError.classList.remove('show');
        updateCameraInfo();
    }
    
    return isValid && empId && name;
}

function updateCameraInfo() {
    const empId = empIdInput.value.trim().toUpperCase();
    const name = nameInput.value.trim();
    
    const empIdContainer = document.getElementById('collectCameraEmpIdContainer');
    const empIdDisplay = document.getElementById('collectCameraEmpId');
    const counterDisplay = document.getElementById('collectCameraCounter');
    
    if (empIdDisplay && empIdContainer) {
        if (empId && validateEmpId(empId)) {
            empIdDisplay.textContent = empId + (name ? '_' + name : '');
            empIdContainer.style.display = 'inline';
        } else {
            empIdContainer.style.display = 'none';
        }
    }
    
    if (counterDisplay) {
        counterDisplay.textContent = `${captureCount}/${MAX_CAPTURES}`;
    }
}

// ============ Camera Functions ============
async function startCamera() {
    try {
        const constraints = {
            video: {
                facingMode: facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.play();
        
        isCameraOn = true;
        toggleCameraBtn.innerHTML = '<i class="fas fa-video-slash"></i> Tắt Camera';
        toggleCameraBtn.classList.remove('collect-btn-primary');
        toggleCameraBtn.classList.add('collect-btn-danger');
        cameraOverlay.classList.remove('show');
        
        updateCaptureButton();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        showToast('Không thể truy cập camera. Vui lòng cấp quyền!', 'error');
        cameraOverlay.innerHTML = `
            <i class="fas fa-camera-slash"></i>
            <p>Không thể truy cập camera</p>
        `;
        cameraOverlay.classList.add('show');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    video.srcObject = null;
    isCameraOn = false;
    toggleCameraBtn.innerHTML = '<i class="fas fa-video"></i> Bật Camera';
    toggleCameraBtn.classList.remove('collect-btn-danger');
    toggleCameraBtn.classList.add('collect-btn-primary');
    cameraOverlay.classList.add('show');
    
    updateCaptureButton();
}

function toggleCamera() {
    if (isCameraOn) {
        stopCamera();
    } else {
        startCamera();
    }
}

async function switchCamera() {
    if (!isCameraOn) {
        showToast('Vui lòng bật camera trước!', 'warning');
        return;
    }
    
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    stopCamera();
    await startCamera();
    
    const cameraType = facingMode === 'user' ? 'trước' : 'sau';
    showToast(`Đã chuyển sang camera ${cameraType}`, 'info');
}

function updateCaptureButton() {
    const isFormValid = validateForm();
    captureBtn.disabled = !isCameraOn || !isFormValid || captureCount >= MAX_CAPTURES;
}

// ============ Progress Functions ============
function updateProgress(count, max = MAX_CAPTURES) {
    captureCount = count;
    const percentage = (count / max) * 100;
    
    progressFill.style.width = `${percentage}%`;
    progressText.textContent = `${count}/${max}`;
    
    progressFill.classList.remove('warning', 'danger');
    if (percentage >= 90) {
        progressFill.classList.add('danger');
    } else if (percentage >= 70) {
        progressFill.classList.add('warning');
    }
    
    updateCameraInfo();
    updateCaptureButton();
}

async function fetchCurrentCount() {
    const empId = empIdInput.value.trim();
    const name = nameInput.value.trim();
    
    if (!empId || !name || !validateEmpId(empId) || !validateName(name)) {
        updateProgress(0);
        return;
    }
    
    try {
        const response = await fetch(`/collect/api/get_count?emp_id=${encodeURIComponent(empId)}&name=${encodeURIComponent(name)}`);
        const data = await response.json();
        
        if (data.success) {
            updateProgress(data.count, data.max);
        }
    } catch (error) {
        console.error('Error fetching count:', error);
    }
}

// ============ Capture Functions ============
async function captureImage() {
    if (!isCameraOn || isCapturing || !validateForm()) {
        return;
    }
    
    if (captureCount >= MAX_CAPTURES) {
        showToast(`Đã đạt giới hạn ${MAX_CAPTURES} ảnh!`, 'warning');
        return;
    }
    
    isCapturing = true;
    captureBtn.disabled = true;
    
    try {
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Flip horizontally
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg', 0.95);
        
        const response = await fetch('/collect/api/capture', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                emp_id: empIdInput.value.trim(),
                name: nameInput.value.trim(),
                image: imageData
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(data.message, 'success');
            updateProgress(data.count, data.max);
            addPreviewImage(imageData);
        } else {
            showToast(data.message, 'error');
        }
        
    } catch (error) {
        console.error('Error capturing image:', error);
        showToast('Lỗi khi chụp ảnh!', 'error');
    }
    
    isCapturing = false;
    updateCaptureButton();
}

function addPreviewImage(imageData) {
    const img = document.createElement('img');
    img.src = imageData;
    img.className = 'collect-preview-image';
    
    previewContainer.insertBefore(img, previewContainer.firstChild);
    
    while (previewContainer.children.length > PREVIEW_MAX) {
        previewContainer.removeChild(previewContainer.lastChild);
    }
}

function clearPreviews() {
    previewContainer.innerHTML = '';
}

// ============ Event Listeners ============
document.addEventListener('DOMContentLoaded', () => {
    toggleCameraBtn.addEventListener('click', toggleCamera);
    switchCameraBtn.addEventListener('click', switchCamera);
    captureBtn.addEventListener('click', captureImage);
    
    empIdInput.addEventListener('input', () => {
        validateForm();
        fetchCurrentCount();
        clearPreviews();
    });
    
    nameInput.addEventListener('input', () => {
        validateForm();
        fetchCurrentCount();
        clearPreviews();
    });
    
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') {
            e.preventDefault();
            captureImage();
        }
    });
    
    cameraOverlay.classList.add('show');
    updateProgress(0);
});

// ============ Cleanup ============
window.addEventListener('beforeunload', () => {
    stopCamera();
});
