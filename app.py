from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import os

app = Flask(__name__)

# Load model
model = load_model('face_emotionModel.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        
        results = []
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(np.expand_dims(face_roi, axis=0), axis=-1)
            
            predictions = model.predict(face_roi, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            emotion = emotion_labels[emotion_idx]
            confidence = float(predictions[0][emotion_idx])
            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{emotion}: {confidence*100:.1f}%", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            results.append({'emotion': emotion, 'confidence': confidence})
        
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'success': True, 'results': results, 'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)