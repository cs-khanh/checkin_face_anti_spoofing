/**
 * Check-in/Check-out page JavaScript functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const videoElement = document.getElementById('video');
    const overlay = document.getElementById('overlay');
    const overlayCtx = overlay.getContext('2d');
    const cameraStatus = document.getElementById('camera-status');
    const actionButtons = document.getElementById('action-buttons');
    const checkInBtn = document.getElementById('check-in-btn');
    const checkOutBtn = document.getElementById('check-out-btn');
    const infoMessage = document.getElementById('info-message');
    const statusIndicator = document.getElementById('status-indicator');
    const recognitionName = document.getElementById('recognition-name');
    const recognitionId = document.getElementById('recognition-id');
    const canvasCapture = document.createElement('canvas');
    const canvasCaptureCtx = canvasCapture.getContext('2d');
    const canvasMotion = document.createElement('canvas');
    const canvasMotionCtx = canvasMotion.getContext('2d');
    
    // Variables
    let currentStream = null;
    let recognizedEmployee = null;
    let lastBbox = null;
    let lastName = null;
    let lastConfidence = null;
    let detectionTimeout = null; // (giữ lại nhưng không dùng nữa)
    let detectionRafId = null;   // CHANGED: id của rAF detect
    let lastFrameData = null;
    const motionThreshold = 0.01;
    let isProcessing = false;
    let isPaused = false;
    let resumeTimer = null;
    // CHANGED: throttle detect
    let lastDetectTime = 0;
    const detectInterval = 150; // ms giữa 2 lần detect (~6-7fps)

    // ================== HÀM QUẢN LÝ PAUSE / RESUME ==================
    function resumeDetection() {
        isPaused = false;
        if (currentStream) {
            try { videoElement.play(); } catch(e) {}
        }
    }

    function pauseDetection(ms) {
        isPaused = true;
        try { videoElement.pause(); } catch(e) {}
        if (resumeTimer) {
            clearTimeout(resumeTimer);
            resumeTimer = null;
        }
        if (typeof ms === 'number' && ms > 0) {
            resumeTimer = setTimeout(() => {
                resumeTimer = null;
                resumeDetection();
            }, ms);
        }
    }
    // ================================================================

    function updateClock() {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        document.getElementById('current-time').textContent = `${hours}:${minutes}:${seconds}`;
        const day = String(now.getDate()).padStart(2, '0');
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const year = now.getFullYear();
        document.getElementById('current-date').textContent = `${day}/${month}/${year}`;
    }
    updateClock();
    setInterval(updateClock, 1000);

    window.addEventListener('beforeunload', function() {
        stopCamera();
    });
    
    startCamera();

    async function startCamera() {
        try {
            const constraints = {
                video: { width: { ideal: 640 }, height: { ideal: 640 }, facingMode: "user" }
            };
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = currentStream;
            videoElement.onloadedmetadata = function() {
                videoElement.play();
                cameraStatus.classList.add('hidden');
                overlay.width = videoElement.videoWidth;
                overlay.height = videoElement.videoHeight;
                // CHANGED: canvasCapture size sẽ set khi capture (downscale)
                canvasMotion.width = Math.floor(videoElement.videoWidth / 4);
                canvasMotion.height = Math.floor(videoElement.videoHeight / 4);
                startDetectionFace();
            };
        } catch (error) {
            console.error('Error accessing camera:', error);
            cameraStatus.innerHTML = `
                <div class="status-message">
                    <i class="bi bi-exclamation-triangle"></i>
                    <span>Không thể truy cập camera. Vui lòng kiểm tra quyền truy cập.</span>
                </div>
            `;
            cameraStatus.classList.add('error');
        }
    }

    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        if (detectionTimeout) {
            clearTimeout(detectionTimeout);
            detectionTimeout = null;
        }
        if (detectionRafId !== null) {            // CHANGED
            cancelAnimationFrame(detectionRafId); // CHANGED
            detectionRafId = null;                // CHANGED
        }
        if (resumeTimer) {
            clearTimeout(resumeTimer);
            resumeTimer = null;
        }
        isPaused = false;
    }

    function calculateMotion() {
        const w = canvasMotion.width;
        const h = canvasMotion.height;
        canvasMotionCtx.save();
        canvasMotionCtx.scale(-1, 1);
        canvasMotionCtx.drawImage(videoElement, 0, 0, videoElement.videoWidth, videoElement.videoHeight, -w, 0, w, h);
        canvasMotionCtx.restore();
        const currentFrameData = canvasMotionCtx.getImageData(0, 0, w, h);
        if (!lastFrameData) {
            lastFrameData = currentFrameData;
            return true;
        }
        // CHANGED: giảm khối lượng tính toán
        let diffPixels = 0;
        const threshold = 10; // 10 -> 15
        const step = 8;       // 4 -> 8
        for (let i = 0; i < currentFrameData.data.length; i += (4 * step)) {
            const rDiff = Math.abs(currentFrameData.data[i] - lastFrameData.data[i]);
            const gDiff = Math.abs(currentFrameData.data[i + 1] - lastFrameData.data[i + 1]);
            const bDiff = Math.abs(currentFrameData.data[i + 2] - lastFrameData.data[i + 2]);
            if (rDiff > threshold || gDiff > threshold || bDiff > threshold) diffPixels++;
        }
        const totalSampledPixels = Math.floor(w * h / step);
        const motionScore = diffPixels / totalSampledPixels;
        lastFrameData = currentFrameData;
        return motionScore > motionThreshold;
    }

    function captureFrame() {
        if (isProcessing || isPaused) return;
        const hasMotion = calculateMotion();
        if (!hasMotion) return;

        // CHANGED: downscale khi gửi server
        const targetW = 640;
        const targetH = 640;
        if (canvasCapture.width !== targetW || canvasCapture.height !== targetH) {
            canvasCapture.width = targetW;
            canvasCapture.height = targetH;
        }

        canvasCaptureCtx.save();
        canvasCaptureCtx.scale(-1, 1);
        canvasCaptureCtx.drawImage(videoElement, -targetW, 0, targetW, targetH);
        canvasCaptureCtx.restore();

        // CHANGED: giảm quality ảnh
        canvasCapture.toBlob(function(blob) {
            detectionFace(blob);
        }, 'image/jpeg', 0.9);
    }

    function startDetectionFace() {
        // CHANGED: dùng rAF + throttle thay cho setTimeout(50)
        function detectLoop(ts) {
            if (!isPaused && (ts - lastDetectTime > detectInterval)) {
                captureFrame();
                lastDetectTime = ts;
            }
            detectionRafId = requestAnimationFrame(detectLoop);
        }
        detectionRafId = requestAnimationFrame(detectLoop);
    }

    function detectionFace(blob) {
        if (isPaused) return;
        isProcessing = true;
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');
        fetch('/face_detect', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {
            if (data.pending === true) {
                lastBbox = data.bbox || null;
                lastName = '⏳ Đang kiểm tra...';
                lastConfidence = 0;
                infoMessage.innerHTML = `
                    <i class="bi bi-hourglass-split"></i>
                    <span>Đang kiểm tra khuôn mặt (${data.window}/5 khung hình)...</span>
                `;
                infoMessage.className = 'alert alert-warning';
                return;
            }

            if (data.success && !data.pending) {
                // CHANGED: chỉ pause khi cần
                if (data.is_real === false || data.warning) {
                    // Spoofing
                    pauseDetection(3000); // dừng 3s cho user đọc cảnh báo
                    console.log(data.fail_reason);
                    lastBbox = data.bbox || null;
                    lastName = '⚠️ FAKE FACE!';
                    lastConfidence = 0;
                    infoMessage.innerHTML = `
                        <i class="bi bi-exclamation-triangle"></i>
                        <span>${data.warning || 'Phát hiện giả mạo! Vui lòng dùng khuôn mặt thật.'}</span>
                        <small class="d-block mt-1">Spoof Score: ${(data.spoof_score * 100).toFixed(1)}%</small>
                    `;
                    infoMessage.className = 'alert alert-danger';
                } else if (data.success && data.bbox && data.confidence > 0.6) {
                    if (data.similarity < 0.9) {
                        // Cần xác nhận → pause tới khi user thao tác
                        pauseDetection();
                        showConfirmationModal(
                            data.employee_name || 'Unknown',
                            data.similarity,
                            () => {
                                lastBbox = data.bbox;
                                lastName = data.employee_name || 'Unknown';
                                lastConfidence = data.similarity;
                                infoMessage.innerHTML = '';
                                infoMessage.className = '';
                                resumeDetection(); // CHANGED: resume ngay khi user xác nhận
                            },
                            () => {
                                lastBbox = null;
                                lastName = null;
                                lastConfidence = null;
                                resumeDetection();
                            }
                        );
                    } else {
                        // Success bình thường: KHÔNG pause nữa để tránh giật
                        lastBbox = data.bbox;
                        lastName = data.employee_name ?? 'Unknown';
                        lastConfidence = data.similarity;
                        if (infoMessage.classList.contains('alert-danger')) {
                            infoMessage.innerHTML = '';
                            infoMessage.className = '';
                        }
                    }
                } else {
                    // Không đạt → không pause để tránh giật; chỉ reset state
                    lastBbox = null;
                    lastName = null;
                    lastConfidence = null;
                }
            } else {
                // !success (server báo fail): không pause
                lastBbox = null;
                lastName = null;
                lastConfidence = null;
                infoMessage.innerHTML = '';
                infoMessage.className = '';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            infoMessage.innerHTML = `
                <i class="bi bi-exclamation-triangle"></i>
                <span>Lỗi kết nối với máy chủ</span>
            `;
            infoMessage.className = 'alert alert-danger';
            // CHANGED: không pause khi lỗi, tránh giật
        })
        .finally(() => {
            isProcessing = false;
        });
    }

    function drawOverlay() {
        if (overlay.width !== videoElement.videoWidth || overlay.height !== videoElement.videoHeight) {
            overlay.width = videoElement.videoWidth;
            overlay.height = videoElement.videoHeight;
        }
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        if (lastBbox) {
            const [x1, y1, x2, y2] = lastBbox.map(v => Math.round(v));
            const bboxWidth = x2 - x1;
            const bboxHeight = y2 - y1;
            const paddingPercent = 0.2;
            const paddingX = bboxWidth * paddingPercent;
            const paddingY = bboxHeight * paddingPercent;
            const x1p = Math.max(x1 - paddingX, 0);
            const y1p = Math.max(y1 - (bboxHeight * (paddingPercent + 0.15)), 0);
            const x2p = Math.min(x2 + paddingX, overlay.width);
            const y2p = Math.min(y2 + (bboxHeight * (paddingPercent + 0.18)), overlay.height);
            const isFake = lastName && lastName.includes('FAKE');
            const isPending = lastName && lastName.includes('Đang kiểm tra');
            const isPausedState = isPaused;
            let boxColor = 'lime';
            if (isFake) boxColor = 'red';
            else if (isPending) boxColor = 'orange';
            else if (isPausedState) boxColor = 'cyan';
            overlayCtx.strokeStyle = boxColor;
            overlayCtx.lineWidth = isFake ? 4 : 3;
            overlayCtx.strokeRect(x1p, y1p, x2p - x1p, y2p - y1p);
            overlayCtx.font = isFake ? 'bold 20px Arial' : '18px Arial';
            overlayCtx.fillStyle = boxColor;
            overlayCtx.fillText(`${lastName} ${(lastConfidence * 100).toFixed(1)}%`, x1p + 4, y1p - 8);
        }
        requestAnimationFrame(drawOverlay);
    }

    function showConfirmationModal(name, similarity, onConfirm, onCancel) {
        const simPct = Math.round((similarity || 0) * 100) / 100;
        let modalEl = document.getElementById('confirm-similarity-modal');
        if (!modalEl) {
            modalEl = document.createElement('div');
            modalEl.id = 'confirm-similarity-modal';
            modalEl.className = 'modal fade';
            modalEl.tabIndex = -1;
            modalEl.innerHTML = `
                <div class="modal-dialog modal-sm modal-dialog-centered">
                    <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Xác nhận danh tính</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <p>Hệ thống nhận dạng: <strong id="confirm-name"></strong></p>
                        <p>Độ tương đồng: <strong id="confirm-sim"></strong></p>
                        <p>Bạn có chắc đó là người này không?</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" id="confirm-no">Không</button>
                        <button type="button" class="btn btn-primary" id="confirm-yes">Có</button>
                    </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modalEl);

            modalEl.querySelector('#confirm-yes').addEventListener('click', () => {
                const bs = modalEl._bsModalInstance;
                if (bs) bs.hide();
                if (typeof onConfirm === 'function') onConfirm();
            });
            modalEl.querySelector('#confirm-no').addEventListener('click', () => {
                const bs = modalEl._bsModalInstance;
                if (bs) bs.hide();
                if (typeof onCancel === 'function') onCancel();
            });

            modalEl.addEventListener('hidden.bs.modal', () => {
                resumeDetection();
            });
        }

        const nameEl = modalEl.querySelector('#confirm-name');
        const simEl = modalEl.querySelector('#confirm-sim');
        if (nameEl) nameEl.textContent = name || 'Unknown';
        if (simEl) simEl.textContent = `${Math.round((similarity || 0) * 100)}%`;

        if (window.bootstrap && typeof window.bootstrap.Modal === 'function') {
            if (!modalEl._bsModalInstance) modalEl._bsModalInstance = new bootstrap.Modal(modalEl, { backdrop: 'static', keyboard: false });
            modalEl._bsModalInstance.show();
        } else {
            const ok = window.confirm(`Xác nhận: ${name}\nĐộ tương đồng: ${Math.round((similarity||0)*100)}%\n\nCó chắc đây là người này không?`);
            if (ok) {
                if (typeof onConfirm === 'function') onConfirm();
            } else {
                if (typeof onCancel === 'function') onCancel();
            }
            resumeDetection();
        }
    }

    requestAnimationFrame(drawOverlay);
});
