

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from ultralytics import YOLO
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time


app = Flask(__name__)
CORS(app)

# Register realtime blueprint
try:
    from realtime import realtime_bp
    app.register_blueprint(realtime_bp)
    print('[app.py] realtime blueprint registered')
except Exception as e:
    print('[app.py] Could not register realtime blueprint:', e)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory store for captions (keyed by saved filename returned to frontend)
VIDEO_CAPTIONS = {}

# Load YOLO model
print("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
print("Model loaded successfully!")

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'main.html')

# Template-based routes
@app.route('/image')
def image_page():
    return render_template('index.html')

@app.route('/video')
def video_page():
    return render_template('video.html')

@app.route('/realtime')
def realtime_page():
    return render_template('realtime.html')


# --- Compatibility: upload & stream endpoints used by templates/video.html ---
@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(video_file.filename)
    timestamp = int(time.time())
    saved_name = f"upload_{timestamp}_{filename}"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
    # Save file under static/uploads
    video_file.save(saved_path)
    # initialize caption entry
    VIDEO_CAPTIONS[saved_name] = ""
    # Return the saved filename (relative) to avoid filesystem absolute-path/URL-encoding issues
    # Frontend will send this name back to /api/video-stream. We will resolve it on the server.
    return jsonify({"video_path": saved_name})


@app.route('/api/video-stream')
def video_stream():
    video_path = request.args.get('video_path')
    if not video_path:
        return "Video not found", 404

    # If frontend sent a relative saved name, resolve it under UPLOAD_FOLDER. If it sent an absolute
    # path (legacy behavior), accept it too.
    if os.path.isabs(video_path):
        real_path = video_path
    else:
        # sanitize the provided filename and resolve
        real_path = os.path.join(UPLOAD_FOLDER, secure_filename(video_path))

    if not os.path.exists(real_path):
        return "Video not found", 404

    def generate():
        cap = cv2.VideoCapture(real_path)
        unique_labels = set()
        frame_idx = 0
        # derive the saved filename key (the frontend uses the saved_name)
        key_name = os.path.basename(real_path)
        # If real_path is absolute but points to a saved file under UPLOAD_FOLDER,
        # prefer the basename as the key. Otherwise key_name will be the saved_name.

        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)[0]
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls_id = r
                label = model.names[int(cls_id)]
                unique_labels.add(label)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {int(score*100)}%', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Compute the running caption (do not draw it on the video frames)
            caption = ("The video contains: " + ", ".join(sorted(unique_labels))) if unique_labels else "No objects detected."

            # Update caption store so the frontend (caption box) can poll and display it.
            if key_name:
                VIDEO_CAPTIONS[key_name] = caption

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            frame_idx += 1

        cap.release()
        try:
            # ensure final caption is saved (in case no frames updated it)
            if key_name and key_name not in VIDEO_CAPTIONS:
                VIDEO_CAPTIONS[key_name] = caption if 'caption' in locals() else ""
            os.remove(real_path)
        except Exception:
            pass

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/video-caption')
def video_caption():
    """Return the running/final caption for a previously uploaded video.

    Query param: video_path (the saved filename returned by /api/upload-video)
    """
    video_path = request.args.get('video_path')
    if not video_path:
        return jsonify({'error': 'video_path required'}), 400

    key = os.path.basename(video_path)
    caption = VIDEO_CAPTIONS.get(key)
    if caption is None or caption == "":
        return jsonify({'status': 'processing', 'caption': ''})
    else:
        labels = [s.strip() for s in caption.replace('The video contains:','').split(',') if s.strip()]
        return jsonify({'status': 'ready', 'caption': caption, 'labels': labels})

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    results = model(img)
    boxes = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        score = float(box.conf[0])
        w, h = x2 - x1, y2 - y1

        boxes.append({
            "x": x1 / img.shape[1],
            "y": y1 / img.shape[0],
            "w": w / img.shape[1],
            "h": h / img.shape[0],
            "label": label,
            "score": round(score, 2)
        })

    # ✅ Add return statement HERE
    unique_labels = list({b["label"] for b in boxes})
    caption = "The image contains " + ", ".join(unique_labels) + "." if unique_labels else "No objects detected."

    return jsonify({"boxes": boxes, "caption": caption})

from flask import Response

@app.route('/api/stream-detect')
def stream_detect():
    def generate():
        cap = cv2.VideoCapture(0)  # Use webcam
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO detection
            results = model(frame)[0]

            # Draw boxes
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls_id = r
                label = model.names[int(cls_id)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {int(score*100)}%', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # Generate a caption
    unique_labels = list({b["label"] for b in boxes})
    caption = "The image contains " + ", ".join(unique_labels) + "." if unique_labels else "No objects detected."

    return jsonify({"boxes": boxes, "caption": caption})


if __name__ == '__main__':
    app.run(debug=True)
