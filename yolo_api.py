from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore, storage

app = Flask(__name__)

# Inisialisasi Firebase Admin SDK menggunakan service account JSON
cred = credentials.Certificate("chilicare-434612-3d5567525c2e.json")  # Pastikan ini adalah path ke file JSON Anda
firebase_admin.initialize_app(cred, {
    'storageBucket': 'chilicare-434612.appspot.com'  # Ganti dengan nama bucket Firebase Storage Anda
})

# Buat koneksi ke Firestore
db = firestore.client()

if not os.path.exists('uploads'):
    os.makedirs('uploads')

model = YOLO('models/yolov8_model.pt')

disease_labels = ['Anthracnose', 'Fail Bloom']

CONFIDENCE_THRESHOLD = 0.3

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_path = os.path.join("uploads", image.filename)
    image.save(image_path)

    results = model(image_path)

    detections = []  
    detected_diseases = []  

    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            confidence = float(box.conf) 

            if confidence >= CONFIDENCE_THRESHOLD:
                class_index = int(box.cls)  

                if class_index < len(disease_labels):
                    disease_name = disease_labels[class_index]  
                else:
                    disease_name = "Unknown Disease" 

                detection_data = {
                    "disease": disease_name,
                    "confidence": confidence,
                }
                detections.append(detection_data)
                detected_diseases.append(disease_name)  

        for detection in detections:
            db.collection('detections').add(detection)

        response = {
            "detected_diseases": detected_diseases,  
            "detections": detections                 
        }
    else:
        
        response = {
            "disease": "No Diseases",  
            "confidence": 0.00,  
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
