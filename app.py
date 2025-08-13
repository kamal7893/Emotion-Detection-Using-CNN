from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
from model import get_model
from yolo_face_detect import detect_faces_yolo
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BEST_MODEL = "LeNet"
model = get_model(BEST_MODEL)
model.load_weights(f"models/{BEST_MODEL}.h5")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion_from_image(image):
    face_boxes = detect_faces_yolo(image)
    if len(face_boxes) == 0:
        faces = face_cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.3, 5)
        face_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in faces]

    for (x1, y1, x2, y2) in face_boxes:
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (48, 48))
        if model.input_shape[-1] == 1:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.expand_dims(face, axis=-1)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        preds = model.predict(face, verbose=0)[0]
        return EMOTIONS[np.argmax(preds)]
    return None

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        emotion = detect_emotion_from_image(frame)
        if emotion:
            cv2.putText(frame, emotion, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    emotion = detect_emotion_from_image(image)
    return render_template('index.html', emotion=emotion if emotion else "No face detected")

if __name__ == '__main__':
    app.run(debug=True)

