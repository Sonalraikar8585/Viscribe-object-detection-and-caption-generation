from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
import time
import subprocess

video_bp = Blueprint('video_bp', __name__)

# Folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'static', 'processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load model once when module is imported
print("[video.py] Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("[video.py] Model loaded")


@video_bp.route('/api/process-video', methods=['POST'])
def process_video():
    """Accepts a form POST with file field 'video'. Returns JSON with URL to processed video.

    Example: frontend posts the file to /api/process-video and receives
    {"status": "success", "url": "/static/processed/processed_<ts>.mp4"}
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded (field name must be "video")'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    saved_name = f"upload_{timestamp}_{filename}"
    input_path = os.path.join(UPLOAD_FOLDER, saved_name)
    file.save(input_path)

    try:
        # Open capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({'error': f'Could not open uploaded video: {input_path}'}), 500

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        out_name = f"processed_{timestamp}.mp4"
        temp_out_path = os.path.join(PROCESSED_FOLDER, f"temp_{timestamp}.avi")
        final_out_path = os.path.join(PROCESSED_FOLDER, out_name)

        # Try to create VideoWriter using a commonly available codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))
            if not out.isOpened():
                cap.release()
                return jsonify({'error': 'Could not open VideoWriter with available codecs.'}), 500

        # Process frames
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference; using stream=True for efficiency
            results = model(frame, stream=True)
            annotated = None
            for r in results:
                # r.plot() returns an annotated frame (numpy array)
                annotated = r.plot()

            if annotated is None:
                out.write(frame)
            else:
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))
                out.write(annotated)

            processed += 1

        cap.release()
        out.release()

        # Convert temporary output to mp4 with libx264 + yuv420p for maximum compatibility, if ffmpeg available
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-i',
            temp_out_path,
            '-vcodec',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            final_out_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                os.remove(temp_out_path)
            except Exception:
                pass
        except Exception:
            try:
                os.rename(temp_out_path, final_out_path)
            except Exception as e:
                if os.path.exists(temp_out_path):
                    os.remove(temp_out_path)
                return jsonify({'error': f'Could not convert or move output file: {e}'}), 500

        # Build URL path that the frontend can use. Assuming the Flask app serves `static/` at '/static/'.
        public_url = os.path.join('/static/processed', out_name)

        return jsonify({'status': 'success', 'url': public_url, 'frames_processed': processed, 'total_frames': frame_count})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@video_bp.route('/api/processed-videos')
def list_processed():
    """Return a list of processed video filenames (simple helper for frontend)."""
    try:
        files = [f for f in os.listdir(PROCESSED_FOLDER) if os.path.isfile(os.path.join(PROCESSED_FOLDER, f))]
        files.sort(reverse=True)
        urls = [os.path.join('/static/processed', f) for f in files]
        return jsonify({'videos': urls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
