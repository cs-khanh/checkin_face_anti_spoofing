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
    const canvasSnapshot = document.createElement('canvas'); // Canvas để lưu snapshot khi pause
    const canvasSnapshotCtx = canvasSnapshot.getContext('2d');
    
    // FIXED: Tạo IMG element để hiển thị snapshot với CÙNG CSS như video
    const snapshotImage = document.createElement('img');
    snapshotImage.id = 'snapshot-overlay';
    snapshotImage.style.cssText = `
        position: absolute;
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) scaleX(-1);
        display: none;
        pointer-events: none;
        z-index: 5;
    `;
    videoElement.parentElement.appendChild(snapshotImage);
    
    // FIXED: Set z-index cho canvas overlay cao hơn snapshot để bounding box hiển thị lên trên
    overlay.style.zIndex = '10';
    
    // Variables
    let currentStream = null;
    let recognizedEmployee = null;
    let lastBbox = null;
    let lastName = null;
    let lastConfidence = null;
    let detectionTimeout = null; // (giữ lại nhưng không dùng nữa)
    let detectionRafId = null;   // CHANGED: id của rAF detect
    let lastFrameData = null;
    const motionThreshold = 0.01; // Ngưỡng phát hiện chuyển động
    let isProcessing = false;
    let isPaused = false;
    let resumeTimer = null;
    // CHANGED: throttle detect
    let lastDetectTime = 0;
    const detectInterval = 80; // ms giữa 2 lần detect (~20fps) - giảm để tăng tốc độ
    let capturedImageData = null; // Lưu ảnh đã capture để dùng cho check-in
    let hasSnapshot = false; // Flag để biết có snapshot hay không
    let isCheckingIn = false; // Flag để prevent double check-in

    // ================== HÀM QUẢN LÝ PAUSE / RESUME ==================
    function resumeDetection() {
        isPaused = false;
        hasSnapshot = false; // Clear snapshot flag
        isCheckingIn = false; // Reset check-in flag
        
        // FIXED: Ẩn snapshot image và hiện video
        snapshotImage.style.display = 'none';
        videoElement.style.display = 'block';
        
        // Reset isProcessing khi resume để cho phép detection tiếp tục
        isProcessing = false;
        
        // Play video lại
        if (currentStream) {
            try { videoElement.play(); } catch(e) {}
        }
        
        // Restart detection loop khi resume
        if (detectionRafId === null) {
            startDetectionFace();
        }
    }

    function pauseDetection(ms) {
        isPaused = true;
        
        // FIXED: Capture snapshot và hiển thị bằng IMG element (cùng CSS như video)
        // Chỉ capture nếu chưa có snapshot (tránh capture lại khi pause nested)
        if (!hasSnapshot && videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
            const vw = videoElement.videoWidth;
            const vh = videoElement.videoHeight;
            canvasSnapshot.width = vw;
            canvasSnapshot.height = vh;
            // KHÔNG mirror vì IMG element sẽ có scaleX(-1) trong CSS
            canvasSnapshotCtx.drawImage(videoElement, 0, 0, vw, vh);
            
            // Convert canvas sang base64 và set làm src của IMG
            const snapshotData = canvasSnapshot.toDataURL('image/jpeg', 0.95);
            snapshotImage.src = snapshotData;
            
            hasSnapshot = true;
        }
        
        // CRITICAL: Luôn ẩn video và hiện snapshot khi pause (ngay cả khi pause lần 2)
        if (hasSnapshot) {
            snapshotImage.style.display = 'block';
            videoElement.style.display = 'none';
        }
        
        // Pause video để tiết kiệm tài nguyên
        try { videoElement.pause(); } catch(e) {}
        
        // Cancel requestAnimationFrame để dừng detection loop hoàn toàn
        if (detectionRafId !== null) {
            cancelAnimationFrame(detectionRafId);
            detectionRafId = null;
        }
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
        hasSnapshot = false;
        // FIXED: Cleanup snapshot image
        snapshotImage.style.display = 'none';
        videoElement.style.display = 'block';
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

        // CHANGED: giảm quality ảnh để tăng tốc độ encode
        canvasCapture.toBlob(function(blob) {
            detectionFace(blob);
        }, 'image/jpeg', 0.7);
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
            // CRITICAL: Kiểm tra isPaused trước khi xử lý response
            // Vì có thể user đã pause trong lúc chờ response
            if (isPaused) {
                console.log('[SKIP] Response received but detection is paused');
                return;
            }
            
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
                    // Capture frame ngay khi detect thành công (trước khi pause)
                    if (data.similarity >= 0.75) {
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = videoElement.videoWidth || 640;
                        tempCanvas.height = videoElement.videoHeight || 480;
                        const tempCtx = tempCanvas.getContext('2d');
                        // Mirror image như ở overlay
                        tempCtx.save();
                        tempCtx.scale(-1, 1);
                        tempCtx.drawImage(videoElement, -tempCanvas.width, 0, tempCanvas.width, tempCanvas.height);
                        tempCtx.restore();
                        // Lưu base64 data
                        capturedImageData = tempCanvas.toDataURL('image/jpeg', 0.75);
                    }
                    
                    if (data.similarity < 0.8) {
                        // Cần xác nhận → modal sẽ tự pause
                        showConfirmationModal(
                            data.employee_name || 'Unknown',
                            data.employee_id || 'Unknown',
                            data.similarity,
                            () => {
                                // FIXED: KHÔNG resume ở đây! Modal đã xử lý logic resume/pause
                                // Chỉ cập nhật thông tin hiển thị
                                lastBbox = data.bbox;
                                lastName = data.employee_name || 'Unknown';
                                lastConfidence = data.similarity;
                                infoMessage.innerHTML = '';
                                infoMessage.className = '';
                            },
                            () => {
                                // User nhấn "Không" → reset thông tin
                                lastBbox = null;
                                lastName = null;
                                lastConfidence = null;
                                // Modal sẽ tự resume khi đóng (qua data-should-resume='true')
                            },
                            data
                        );
                    } else {
                        // Success cao → check-in luôn không cần hỏi
                        // COMMENTED: Nếu muốn hỏi xác nhận trước khi check-in, uncomment phần dưới
                        /*
                        pauseDetection();
                        showCheckInModal(
                            data.employee_name || 'Unknown',
                            data.employee_id || 'Unknown',
                            data.similarity,
                            data
                        );
                        */
                        
                        // Check-in trực tiếp khi similarity >= 0.8
                        lastBbox = data.bbox;
                        lastName = data.employee_name ?? 'Unknown';
                        lastConfidence = data.similarity;
                        if (infoMessage.classList.contains('alert-danger')) {
                            infoMessage.innerHTML = '';
                            infoMessage.className = '';
                        }
                        // Set isProcessing để block các capture mới ngay lập tức
                        isProcessing = true;
                        // Gọi performCheckIn trực tiếp
                        performCheckIn(data.employee_name, data.employee_id, data.similarity);
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
            // Chỉ reset isProcessing nếu không bị pause
            // Nếu đã pause, giữ isProcessing = true để ngăn capture frame mới
            if (!isPaused) {
                isProcessing = false;
            }
        });
    }

    function drawOverlay() {
        if (overlay.width !== videoElement.videoWidth || overlay.height !== videoElement.videoHeight) {
            overlay.width = videoElement.videoWidth;
            overlay.height = videoElement.videoHeight;
        }
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        
        // FIXED: KHÔNG vẽ snapshot lên overlay canvas nữa
        // Snapshot được hiển thị bằng IMG element có cùng CSS như video
        
        if (lastBbox) {
            const [x1, y1, x2, y2] = lastBbox.map(v => Math.round(v));
            const bboxWidth = x2 - x1;
            const bboxHeight = y2 - y1;
            const paddingPercent = 0.2;
            const paddingX = bboxWidth * paddingPercent;
            const paddingY = bboxHeight * paddingPercent;
            const x1p = Math.max(x1 - paddingX, 0);
            const y1p = Math.max(y1 - (bboxHeight * (paddingPercent + 0.10)), 0);
            const x2p = Math.min(x2 + paddingX, overlay.width);
            const y2p = Math.min(y2 + (bboxHeight * (paddingPercent + 0.13)), overlay.height);
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

    function showConfirmationModal(name, empCode, similarity, onConfirm, onCancel, fullData) {
        // Pause detection ngay lập tức khi modal hiện lên
        pauseDetection();
        
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
                        <p>Mã NV: <strong id="confirm-empcode"></strong></p>
                        <p>Độ tương đồng: <strong id="confirm-sim"></strong></p>
                        <p>Bạn có chắc đó là người này không?</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" id="confirm-no">Không</button>
                        <button type="button" class="btn btn-primary" id="confirm-yes-checkin">Có, Check-in</button>
                    </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modalEl);

            modalEl.addEventListener('hidden.bs.modal', () => {
                // CHỈ resume nếu user nhấn "Không", không resume nếu nhấn "Có, Check-in"
                // Kiểm tra attribute để quyết định
                if (modalEl.getAttribute('data-should-resume') === 'true') {
                    resumeDetection();
                }
                // Reset attribute cho lần sau
                modalEl.removeAttribute('data-should-resume');
            });
        }
        
        // Setup button handlers cho lần show này
        const yesBtn = modalEl.querySelector('#confirm-yes-checkin');
        const noBtn = modalEl.querySelector('#confirm-no');
        
        // Clone và replace để xóa old listeners
        const newYesBtn = yesBtn.cloneNode(true);
        const newNoBtn = noBtn.cloneNode(true);
        yesBtn.parentNode.replaceChild(newYesBtn, yesBtn);
        noBtn.parentNode.replaceChild(newNoBtn, noBtn);
        
        newYesBtn.addEventListener('click', () => {
            // FIXED: Đánh dấu KHÔNG resume khi đóng modal (vì sẽ check-in)
            modalEl.setAttribute('data-should-resume', 'false');
            const bs = modalEl._bsModalInstance;
            if (bs) bs.hide();
            if (typeof onConfirm === 'function') onConfirm();
            // Call check-in after confirmation
            if (fullData) {
                performCheckIn(fullData.employee_name, fullData.employee_id, fullData.similarity);
            }
        });
        
        newNoBtn.addEventListener('click', () => {
            // User từ chối → cho phép resume
            modalEl.setAttribute('data-should-resume', 'true');
            const bs = modalEl._bsModalInstance;
            if (bs) bs.hide();
            if (typeof onCancel === 'function') onCancel();
        });

        const nameEl = modalEl.querySelector('#confirm-name');
        const empCodeEl = modalEl.querySelector('#confirm-empcode');
        const simEl = modalEl.querySelector('#confirm-sim');
        if (nameEl) nameEl.textContent = name || 'Unknown';
        if (empCodeEl) empCodeEl.textContent = empCode || 'Unknown';
        if (simEl) simEl.textContent = `${Math.round((similarity || 0) * 100)}%`;

        if (window.bootstrap && typeof window.bootstrap.Modal === 'function') {
            if (!modalEl._bsModalInstance) modalEl._bsModalInstance = new bootstrap.Modal(modalEl, { backdrop: 'static', keyboard: false });
            modalEl._bsModalInstance.show();
        } else {
            const ok = window.confirm(`Xác nhận: ${name} (${empCode})\nĐộ tương đồng: ${Math.round((similarity||0)*100)}%\n\nCó chắc đây là người này không?`);
            if (ok) {
                if (typeof onConfirm === 'function') onConfirm();
                if (fullData) {
                    // FIXED: Không resume khi check-in, performCheckIn sẽ xử lý pause
                    performCheckIn(fullData.employee_name, fullData.employee_id, fullData.similarity);
                }
            } else {
                if (typeof onCancel === 'function') onCancel();
                // CHỈ resume khi user nhấn Cancel/Không
                resumeDetection();
            }
        }
    }

    function showCheckInModal(name, empCode, similarity, fullData) {
        let modalEl = document.getElementById('checkin-modal');
        if (!modalEl) {
            modalEl = document.createElement('div');
            modalEl.id = 'checkin-modal';
            modalEl.className = 'modal fade';
            modalEl.tabIndex = -1;
            modalEl.innerHTML = `
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                    <div class="modal-header bg-primary text-white">
                        <h5 class="modal-title"><i class="bi bi-person-check"></i> Xác nhận Check-in</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body text-center">
                        <i class="bi bi-person-circle" style="font-size: 4rem; color: #0d6efd;"></i>
                        <h4 class="mt-3" id="checkin-name"></h4>
                        <p class="text-muted mb-1">Mã NV: <strong id="checkin-empcode"></strong></p>
                        <p class="text-muted">Độ chính xác: <strong id="checkin-sim"></strong></p>
                        <hr>
                        <p>Bạn muốn check-in?</p>
                    </div>
                    <div class="modal-footer justify-content-center">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Hủy</button>
                        <button type="button" class="btn btn-success" id="checkin-confirm-btn">
                            <i class="bi bi-check-circle"></i> Check-in
                        </button>
                    </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modalEl);

            modalEl.querySelector('#checkin-confirm-btn').addEventListener('click', () => {
                const bs = modalEl._bsModalInstance;
                if (bs) bs.hide();
                performCheckIn(fullData.employee_name, fullData.employee_id, fullData.similarity);
            });

            modalEl.addEventListener('hidden.bs.modal', () => {
                resumeDetection();
            });
        }

        const nameEl = modalEl.querySelector('#checkin-name');
        const empCodeEl = modalEl.querySelector('#checkin-empcode');
        const simEl = modalEl.querySelector('#checkin-sim');
        if (nameEl) nameEl.textContent = name || 'Unknown';
        if (empCodeEl) empCodeEl.textContent = empCode || 'Unknown';
        if (simEl) simEl.textContent = `${Math.round((similarity || 0) * 100)}%`;

        if (window.bootstrap && typeof window.bootstrap.Modal === 'function') {
            if (!modalEl._bsModalInstance) modalEl._bsModalInstance = new bootstrap.Modal(modalEl, { backdrop: 'static', keyboard: false });
            modalEl._bsModalInstance.show();
        } else {
            const ok = window.confirm(`Check-in cho: ${name} (${empCode})?`);
            if (ok) {
                performCheckIn(fullData.employee_name, fullData.employee_id, fullData.similarity);
            }
            resumeDetection();
        }
    }

    function performCheckIn(employeeName, empCode, matchScore) {
        // FIXED: Prevent double check-in
        if (isCheckingIn) {
            console.log('Check-in already in progress, skipping...');
            return;
        }
        isCheckingIn = true;
        
        // Pause detection ngay lập tức để dừng mọi API call
        pauseDetection();
        
        // Dùng ảnh đã capture từ biến toàn cục (capture ngay khi detect)
        const imageData = capturedImageData || '';
        
        fetch('/api/checkin', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                emp_code: empCode,
                employee_name: employeeName,
                match_score: matchScore,
                image: imageData
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Hiển thị modal thành công thay vì toast
                showCheckinSuccessModal(data.data || {
                    employee_name: employeeName,
                    emp_code: empCode,
                    event_time: new Date().toISOString()
                });
                // isCheckingIn sẽ được reset khi resume (sau khi đóng modal)
            } else {
                showErrorToast(`❌ ${data.message}`);
                isCheckingIn = false; // Reset flag
                resumeDetection();
            }
        })
        .catch(error => {
            console.error('Check-in error:', error);
            showErrorToast('❌ Lỗi kết nối với máy chủ');
            isCheckingIn = false; // Reset flag
            resumeDetection();
        });
    }

    function showSuccessToast(message) {
        const toastHtml = `
            <div class="toast align-items-center text-white bg-success border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        showToast(toastHtml);
    }

    function showErrorToast(message) {
        const toastHtml = `
            <div class="toast align-items-center text-white bg-danger border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        showToast(toastHtml);
    }

    function showToast(toastHtml) {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = toastHtml;
        const toastEl = tempDiv.firstElementChild;
        container.appendChild(toastEl);
        
        if (window.bootstrap && typeof window.bootstrap.Toast === 'function') {
            const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
            toast.show();
            toastEl.addEventListener('hidden.bs.toast', () => {
                toastEl.remove();
            });
        } else {
            alert(toastEl.querySelector('.toast-body').textContent);
            toastEl.remove();
        }
    }

    function showCheckinSuccessModal(data) {
        let modalEl = document.getElementById('checkin-success-modal');
        if (!modalEl) {
            modalEl = document.createElement('div');
            modalEl.id = 'checkin-success-modal';
            modalEl.className = 'modal fade';
            modalEl.tabIndex = -1;
            modalEl.innerHTML = `
                <div class="modal-dialog modal-dialog-centered">
                    <div class="modal-content">
                        <div class="modal-header bg-success text-white">
                            <h5 class="modal-title">
                                <i class="bi bi-check-circle-fill"></i> Check-in thành công!
                            </h5>
                        </div>
                        <div class="modal-body text-center py-4">
                            <div class="mb-3">
                                <i class="bi bi-person-check-fill text-success" style="font-size: 5rem;"></i>
                            </div>
                            <h3 class="text-success mb-3" id="success-name"></h3>
                            <div class="text-start mx-auto" style="max-width: 350px;">
                                <p class="mb-2">
                                    <strong>Mã nhân viên:</strong> 
                                    <span id="success-empcode" class="text-primary"></span>
                                </p>
                                <p class="mb-2">
                                    <strong>Thời gian:</strong> 
                                    <span id="success-time"></span>
                                </p>
                                <p class="mb-2">
                                    <strong>Ngày làm việc:</strong> 
                                    <span id="success-date"></span>
                                </p>
                                <p class="mb-0">
                                    <strong>Độ chính xác:</strong> 
                                    <span id="success-score" class="badge bg-success"></span>
                                </p>
                            </div>
                        </div>
                        <div class="modal-footer justify-content-center">
                            <button type="button" class="btn btn-primary btn-lg" data-bs-dismiss="modal">
                                <i class="bi bi-x-circle"></i> Đóng
                            </button>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(modalEl);

            modalEl.addEventListener('hidden.bs.modal', () => {
                // Đợi 2 giây sau khi đóng modal mới resume detection
                setTimeout(() => {
                    resumeDetection();
                }, 2000);
            });
        }

        // Update modal content
        const nameEl = modalEl.querySelector('#success-name');
        const empCodeEl = modalEl.querySelector('#success-empcode');
        const timeEl = modalEl.querySelector('#success-time');
        const dateEl = modalEl.querySelector('#success-date');
        const scoreEl = modalEl.querySelector('#success-score');

        if (nameEl) nameEl.textContent = data.employee_name || 'Unknown';
        if (empCodeEl) empCodeEl.textContent = data.emp_code || 'Unknown';
        
        if (timeEl) {
            const eventTime = data.event_time ? new Date(data.event_time) : new Date();
            timeEl.textContent = eventTime.toLocaleTimeString('vi-VN');
        }
        
        if (dateEl) {
            const workDate = data.work_date ? new Date(data.work_date) : new Date();
            dateEl.textContent = workDate.toLocaleDateString('vi-VN');
        }
        
        if (scoreEl) {
            const score = Math.round((data.match_score || 0) * 100);
            scoreEl.textContent = `${score}%`;
        }

        // Show modal
        if (window.bootstrap && typeof window.bootstrap.Modal === 'function') {
            if (!modalEl._bsModalInstance) {
                modalEl._bsModalInstance = new bootstrap.Modal(modalEl, { 
                    backdrop: 'static', 
                    keyboard: false 
                });
            }
            modalEl._bsModalInstance.show();
        } else {
            alert(`✅ Check-in thành công!\n${data.employee_name} (${data.emp_code})`);
            setTimeout(() => {
                resumeDetection();
            }, 2000);
        }
    }

    requestAnimationFrame(drawOverlay);
});
