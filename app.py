from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from ultralytics import YOLO
import cv2
import os
from datetime import datetime

app = Flask(__name__)

# === Load model once (important!) ===
model_path = "models/last.pt"
model = YOLO(model_path)
class_names = model.names

# === Directories ===
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return 'No file uploaded', 400

    filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))
    output_filename = "result_" + filename
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    chicken_count = 0

    if is_video:
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        unique_ids = set()

        for results in model.track(source=input_path, persist=True, stream=True, conf=0.25, tracker="bytetrack.yaml"):
            frame = results.orig_img.copy()
            boxes = results.boxes

            if boxes.id is not None:
                for box, track_id in zip(boxes, boxes.id):
                    track_id = int(track_id)
                    unique_ids.add(track_id)

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    label = f"{class_names[cls_id]} - ID {track_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(frame)

        cap.release()
        out.release()
        chicken_count = len(unique_ids)

    else:
        results = model(input_path)[0]
        image = results.orig_img.copy()
        chicken_count = len(results.boxes)

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = f"{class_names[cls_id]} - ID {i+1}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(output_path, image)

    return render_template("result.html",
                           result_file=output_filename,
                           is_video=is_video,
                           chicken_count=chicken_count)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
