from flask import Blueprint, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64

realtime_bp = Blueprint('realtime_bp', __name__)

# Load a model instance for real-time detection
print('[realtime.py] Loading YOLOv8 model for realtime...')
model = YOLO('yolov8n.pt')
print('[realtime.py] Model loaded')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')


def _run_model(img, conf=0.25):
    """Run model with a given confidence threshold, with a safe fallback."""
    try:
        results = model(img, conf=conf)
    except TypeError:
        # older/newer ultralytics API differences
        results = model(img)
    return results


def _parse_results(results, img_shape):
    boxes = []
    for box in results[0].boxes:
        try:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
        except Exception:
            # fallback if xyxy is already a flat list
            vals = box.xyxy
            x1, y1, x2, y2 = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        try:
            cls_id = int(box.cls[0])
        except Exception:
            try:
                cls_id = int(box.cls)
            except Exception:
                cls_id = 0
        try:
            conf = float(box.conf[0])
        except Exception:
            try:
                conf = float(box.conf)
            except Exception:
                conf = 0.0

        label = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)
        w, h = x2 - x1, y2 - y1
        boxes.append({
            'x': x1 / img_shape[1],
            'y': y1 / img_shape[0],
            'w': w / img_shape[1],
            'h': h / img_shape[0],
            'label': label,
            'score': round(conf, 2)
        })
    return boxes


@realtime_bp.route('/api/realtime-detect', methods=['POST'])
def realtime_detect():
    """Return boxes and caption for an uploaded frame (no annotated image)."""
    # Accept multipart file 'frame' or raw body bytes
    if 'frame' in request.files:
        frame_file = request.files['frame']
        data = frame_file.read()
    else:
        data = request.get_data()
        if not data:
            return jsonify({'error': 'No frame uploaded'}), 400

    nparr = np.frombuffer(data, np.uint8)
    # defensive checks
    if nparr.size == 0:
        print('[realtime-frame] empty buffer received; request.files keys=', list(request.files.keys()), 'content_length=', request.content_length)
        return jsonify({'error': 'Empty frame buffer received'}), 400

    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except cv2.error as e:
        print('[realtime-frame] cv2.imdecode raised cv2.error:', e)
        return jsonify({'error': 'Failed to decode image (cv2.imdecode error)'}), 400

    if img is None:
        print('[realtime-frame] imdecode returned None; buffer len=', nparr.size)
        return jsonify({'error': 'Failed to decode image (imdecode returned None)'}), 400

    try:
        results = _run_model(img, conf=0.25)
        boxes = _parse_results(results, img.shape)
        unique_labels = list({b['label'] for b in boxes})
        caption = "The frame contains " + ", ".join(unique_labels) + "." if unique_labels else "No objects detected."

        # Debug log
        try:
            print(f'[realtime-detect] frame={img.shape} detections={len(boxes)} labels={unique_labels}')
        except Exception:
            pass

        return jsonify({'boxes': boxes, 'caption': caption, 'labels': unique_labels, 'count': len(boxes)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@realtime_bp.route('/api/realtime-status')
def realtime_status():
    return jsonify({'status': 'ok'})


@realtime_bp.route('/api/realtime-frame', methods=['POST'])
def realtime_frame():
    """Return annotated frame (base64), boxes and caption for display in frontend."""
    if 'frame' in request.files:
        data = request.files['frame'].read()
    else:
        data = request.get_data()
        if not data:
            return jsonify({'error': 'No frame uploaded'}), 400

    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Failed to decode image'}), 400

    try:
        results = _run_model(img, conf=0.25)
        boxes = _parse_results(results, img.shape)
        unique_labels = list({b['label'] for b in boxes})
        caption = "The frame contains " + ", ".join(unique_labels) + "." if unique_labels else "No objects detected."

        # Create annotated frame using ultralytics plotting helper
        try:
            annotated = results[0].plot()
        except Exception:
            # fallback to drawing boxes manually if .plot() not available
            annotated = img.copy()
            for b in boxes:
                x = int(b['x'] * annotated.shape[1])
                y = int(b['y'] * annotated.shape[0])
                w = int(b['w'] * annotated.shape[1])
                h = int(b['h'] * annotated.shape[0])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, f"{b['label']} {int(b['score']*100)}%", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        _, buff = cv2.imencode('.jpg', annotated)
        jpg_bytes = buff.tobytes()
        b64 = base64.b64encode(jpg_bytes).decode('utf-8')
        data_url = f'data:image/jpeg;base64,{b64}'

        # Debug log
        try:
            print(f'[realtime-frame] frame={img.shape} detections={len(boxes)} labels={unique_labels}')
        except Exception:
            pass

        return jsonify({'annotated': data_url, 'caption': caption, 'boxes': boxes, 'labels': unique_labels, 'count': len(boxes)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
