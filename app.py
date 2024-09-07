from flask import Flask, request, jsonify
from flask_cors import CORS  # Importa o CORS
import numpy as np
import cv2
from pre_processamento import preprocess_image, extract_features
import joblib

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

# Carregar o modelo treinado
model = joblib.load('svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    if file:
        # Salvar a imagem temporariamente
        image_path = 'temp_image.png'
        file.save(image_path)
        
        # Processar a imagem
        img_bin = preprocess_image(image_path)
        features = extract_features(img_bin)
        features = np.array([features])  # Transformar em array numpy
        
        # Fazer a previsão
        diagnosis = model.predict(features)[0]
        
        # Mapear diagnóstico
        if diagnosis in [0, 1]:
            result = "Não detectado predisposição ao Alzheimer ou Parkinson"
        elif diagnosis == 2:
            result = "Predisposição a Alzheimer ou Parkinson"
        else:
            result = "Diagnóstico desconhecido"
        
        return jsonify({'diagnosis': result})

    return jsonify({'error': 'Image processing failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
