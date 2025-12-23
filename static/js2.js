const video = document.getElementById('liveVideo');
const canvas = document.getElementById('overlayCanvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const captionArea = document.getElementById('captionArea');
const ttsBtn = document.getElementById('ttsBtn');

let stream = null;
let interval = null;
let currentCaption = "";

// Start camera
async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  startBtn.disabled = true;
  stopBtn.disabled = false;
  ttsBtn.disabled = true;

  interval = setInterval(processFrame, 500);
}

// Stop camera
function stopCamera() {
  clearInterval(interval);
  if (stream) stream.getTracks().forEach(track => track.stop());
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

// Send frame to backend
async function processFrame() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  tempCanvas.getContext('2d').drawImage(video, 0, 0);

  const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg'));

  const formData = new FormData();
  formData.append('frame', blob, 'frame.jpg');

  try {
    const response = await fetch('/api/realtime-detect', { method: 'POST', body: formData });
    const result = await response.json();
    drawDetections(result);
  } catch (err) {
    console.error("Detection error:", err);
  }
}

// Draw boxes
function drawDetections(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (!result || !result.boxes) return;

  ctx.lineWidth = 2;
  ctx.font = '16px Arial';

  result.boxes.forEach(b => {
    const x = b.x * canvas.width;
    const y = b.y * canvas.height;
    const w = b.w * canvas.width;
    const h = b.h * canvas.height;
    ctx.strokeStyle = '#ff0000';
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = '#ff0000';
    ctx.fillText(`${b.label} ${(b.score*100).toFixed(0)}%`, x + 4, y - 6);
  });

  currentCaption = result.caption;
  captionArea.textContent = currentCaption;
  ttsBtn.disabled = !currentCaption;
}

// Text-to-speech
ttsBtn.addEventListener('click', () => {
  if (!currentCaption) return;
  const utter = new SpeechSynthesisUtterance(currentCaption);
  utter.lang = 'en-US';
  window.speechSynthesis.speak(utter);
});

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
