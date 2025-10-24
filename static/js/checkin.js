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
    // Canvas ẩn để gửi frame lên server
    const canvasCapture = document.createElement('canvas');
    const canvasCaptureCtx = canvasCapture.getContext('2d');
    
    // Canvas để phát hiện motion
    const canvasMotion = document.createElement('canvas');
    const canvasMotionCtx = canvasMotion.getContext('2d');
    
    // Variables
    let currentStream = null;
    let recognizedEmployee = null;
    let lastBbox = null;
    let lastName = null;
    let lastConfidence = null;
    let detectionTimeout = null;
    let lastFrameData = null;
    const motionThreshold = 0.03; // 3% thay đổi pixel (nhạy hơn)
    let isProcessing = false; // Flag để tránh gọi API trùng lặp
    
    // Update clock
    function updateClock() {
        const now = new Date();
        
        // Format time: HH:MM:SS
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        document.getElementById('current-time').textContent = `${hours}:${minutes}:${seconds}`;
        
        // Format date: DD/MM/YYYY
        const day = String(now.getDate()).padStart(2, '0');
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const year = now.getFullYear();
        document.getElementById('current-date').textContent = `${day}/${month}/${year}`;
    }
    updateClock();
    setInterval(updateClock, 1000);


    // // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        stopCamera();
    });
    
    // Start the camera on page load
    startCamera();
    // // Initialize the camera
    async function startCamera() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 480 },  // Giảm từ 640 -> 480 để faster
                    height: { ideal: 480 },
                    facingMode: "user"
                }
            };
            
            // Get user media
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = currentStream;
            
            // Wait for the video to be ready
            videoElement.onloadedmetadata = function() {
                videoElement.play();
                cameraStatus.classList.add('hidden');
                
                // Canvas setup
                overlay.width = videoElement.videoWidth;
                overlay.height = videoElement.videoHeight;
                canvasCapture.width = videoElement.videoWidth;
                canvasCapture.height = videoElement.videoHeight;
                // Motion canvas dùng resolution thấp hơn để tính nhanh hơn
                canvasMotion.width = Math.floor(videoElement.videoWidth / 4);
                canvasMotion.height = Math.floor(videoElement.videoHeight / 4);

                // Start face detection
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
    
    // Stop the camera
    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => {
                track.stop();
            });
            currentStream = null;
        }
        
            if (detectionTimeout) {
                clearTimeout(detectionTimeout);
                detectionTimeout = null;
            }
    }
    
    // Tính toán mức độ chuyển động giữa 2 frame (optimized)
    function calculateMotion() {
        const w = canvasMotion.width;
        const h = canvasMotion.height;
        
        // Vẽ frame hiện tại lên canvas motion với resolution thấp (lật ngang)
        canvasMotionCtx.save();
        canvasMotionCtx.scale(-1, 1);
        canvasMotionCtx.drawImage(videoElement, 0, 0, videoElement.videoWidth, videoElement.videoHeight, 
                                   -w, 0, w, h);
        canvasMotionCtx.restore();
        
        const currentFrameData = canvasMotionCtx.getImageData(0, 0, w, h);
        
        // Nếu chưa có frame trước, lưu lại và return true (để gọi API lần đầu)
        if (!lastFrameData) {
            lastFrameData = currentFrameData;
            return true;
        }
        
        // So sánh pixel giữa 2 frame (tối ưu: bỏ qua một số pixel)
        let diffPixels = 0;
        const threshold = 10; // ngưỡng khác biệt RGB
        const step = 4; // Skip pixels để tính nhanh hơn
        
        for (let i = 0; i < currentFrameData.data.length; i += (4 * step)) {
            const rDiff = Math.abs(currentFrameData.data[i] - lastFrameData.data[i]);
            const gDiff = Math.abs(currentFrameData.data[i + 1] - lastFrameData.data[i + 1]);
            const bDiff = Math.abs(currentFrameData.data[i + 2] - lastFrameData.data[i + 2]);
            
            if (rDiff > threshold || gDiff > threshold || bDiff > threshold) {
                diffPixels++;
            }
        }
        
        // Tính % pixel thay đổi
        const totalSampledPixels = Math.floor(w * h / step);
        const motionScore = diffPixels / totalSampledPixels;
        
        // Lưu frame hiện tại để so sánh lần sau
        lastFrameData = currentFrameData;
        
        return motionScore > motionThreshold;
    }
    
    // // // Capture a frame from the video
    function captureFrame() {
        // Bỏ qua nếu đang xử lý request trước đó
        if (isProcessing) {
            return;
        }
        
        // Kiểm tra motion trước
        const hasMotion = calculateMotion();
        
        if (!hasMotion) {
            return; // Không có chuyển động đáng kể, bỏ qua
        }
        
        // Đảm bảo canvasCapture luôn đúng kích thước video
        if (canvasCapture.width !== videoElement.videoWidth || canvasCapture.height !== videoElement.videoHeight) {
            canvasCapture.width = videoElement.videoWidth;
            canvasCapture.height = videoElement.videoHeight;
        }
        const w = canvasCapture.width;
        const h = canvasCapture.height;
        // Lật ngang khi capture
        canvasCaptureCtx.save();
        canvasCaptureCtx.scale(-1, 1);
        canvasCaptureCtx.drawImage(videoElement, -w, 0, w, h);
        canvasCaptureCtx.restore();
        // Convert to blob và gửi lên server
        canvasCapture.toBlob(function(blob) {
            detectionFace(blob);
        }, 'image/jpeg', 0.75); // Giảm quality xuống 0.75 để nhanh hơn (balance quality/speed)
    }
    
    // // // Start face detection process
    function startDetectionFace() {
            function detectLoop() {
                captureFrame();
                // Sử dụng requestAnimationFrame để smooth hơn, fallback setTimeout
                detectionTimeout = setTimeout(detectLoop, 33); // 33ms = ~30 FPS (real-time)
            }
            detectLoop();
    }

    // Send the frame to the backend for face detection
    function detectionFace(blob) {
        isProcessing = true;
        
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');
        fetch('/face_detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Check for spoofing detection
            if (data.is_real === false || data.warning) {
                // Spoofing detected!
                lastBbox = data.bbox || null;
                lastName = '⚠️ FAKE FACE!';
                lastConfidence = 0;
                
                // Show warning message
                infoMessage.innerHTML = `
                    <i class="bi bi-exclamation-triangle"></i>
                    <span>${data.warning || 'Phát hiện giả mạo! Vui lòng dùng khuôn mặt thật.'}</span>
                    <small class="d-block mt-1">Spoof Score: ${(data.spoof_score * 100).toFixed(1)}%</small>
                `;
                infoMessage.className = 'alert alert-danger';
            } else if (data.success && data.bbox && data.confidence > 0.6) {
                // Valid real face
                lastBbox = data.bbox;
                lastName = data.employee_name ?? 'Unknown';
                lastConfidence = data.similarity;
                
                // Clear warning if any
                if (infoMessage.classList.contains('alert-danger')) {
                    infoMessage.innerHTML = '';
                    infoMessage.className = '';
                }
            } else {
                lastBbox = null;
                lastName = null;
                lastConfidence = null;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            infoMessage.innerHTML = `
                <i class="bi bi-exclamation-triangle"></i>
                <span>Lỗi kết nối với máy chủ</span>
            `;
            infoMessage.className = 'alert alert-danger';
        })
        .finally(() => {
            isProcessing = false; // Cho phép gọi API tiếp theo
        });
    }
    
    // Vẽ overlay bbox mượt mà
    function drawOverlay() {
        // Luôn resize overlay đúng với video
        if (overlay.width !== videoElement.videoWidth || overlay.height !== videoElement.videoHeight) {
            overlay.width = videoElement.videoWidth;
            overlay.height = videoElement.videoHeight;
        }
        overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        if (lastBbox) {
            const [x1, y1, x2, y2] = lastBbox.map(v => Math.round(v));
            const bboxWidth = x2 - x1;
            const bboxHeight = y2 - y1;
            const paddingPercent = 0.1; // 10%
            const paddingX = bboxWidth * paddingPercent;
            const paddingY = bboxHeight * paddingPercent;

            // Mở rộng khung ra ngoài
            const x1p = Math.max(x1 - paddingX, 0);
            const y1p = Math.max(y1 - (bboxHeight * (paddingPercent + 0.2)), 0);
            const x2p = Math.min(x2 + paddingX, overlay.width);
            const y2p = Math.min(y2 + (bboxHeight * (paddingPercent + 0.23)), overlay.height);

            // Màu sắc: đỏ nếu fake face, xanh nếu real face
            const isFake = lastName && lastName.includes('FAKE');
            const boxColor = isFake ? 'red' : 'lime';
            
            overlayCtx.strokeStyle = boxColor;
            overlayCtx.lineWidth = isFake ? 4 : 3; // Dày hơn nếu fake
            overlayCtx.strokeRect(x1p, y1p, x2p - x1p, y2p - y1p);
            overlayCtx.font = isFake ? 'bold 20px Arial' : '18px Arial';
            overlayCtx.fillStyle = boxColor;
            overlayCtx.fillText(
                `${lastName} ${(lastConfidence * 100).toFixed(1)}%`,
                x1p + 4, y1p - 8
            );
        }
        requestAnimationFrame(drawOverlay);
    }
    // Start overlay drawing
    requestAnimationFrame(drawOverlay);
    
    
    // // // When modal is hidden, resume capturing
    // // document.getElementById('successModal').addEventListener('hidden.bs.modal', function() {
    // //     if (!captureInterval) {
    // //         document.getElementById('restart-scan-btn').classList.remove('d-none');
    // //     }
    // // });
    
});