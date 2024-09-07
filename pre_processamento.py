import os
import cv2
import numpy as np

# Função para pré-processar as imagens: redimensionar e converter para tons de cinza
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Carrega a imagem em tons de cinza
    img_resized = cv2.resize(img, (128, 128))  # Redimensiona para um tamanho fixo
    _, img_bin = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY)  # Binariza a imagem
    return img_bin

# Função para detectar círculos na imagem
def detect_circle(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=60)
    return 1 if circles is not None else 0

# Função para detectar linhas (ponteiros) na imagem
def detect_lines(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    correct_pointers = 0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if 10 < angle < 20:  # Verifica o ângulo aproximado para o ponteiro maior (2 horas)
                correct_pointers += 1
            elif 70 < angle < 80:  # Verifica o ângulo aproximado para o ponteiro menor (11 horas)
                correct_pointers += 1
    return correct_pointers == 2

# Função para detectar números (baseado na presença de contornos)
def detect_numbers(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    return 1 if num_contours >= 12 else 0  # Espera-se que haja pelo menos 12 contornos representando números

# Função para extrair características importantes da imagem
def extract_features(img):
    white_pixels = np.sum(img == 255)  # Conta os pixels brancos
    black_pixels = np.sum(img == 0)  # Conta os pixels pretos
    
    symm_feature = np.sum(np.abs(img[:, :64] - img[:, 64:]))  # Simetria
    
    has_circle = detect_circle(img)  # Detecta o círculo
    has_numbers = detect_numbers(img)  # Detecta números
    has_correct_pointers = detect_lines(img)  # Detecta os ponteiros corretos

    return [white_pixels, black_pixels, symm_feature, has_circle, has_numbers, has_correct_pointers]

# Função para carregar dados e processar imagens
def load_and_preprocess_data(dataset_path='./dataset/'):
    feature_vectors = []
    labels = []

    class_mapping = {'perfeito': 0, 'medio': 1, 'ruim': 2}

    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                img_bin = preprocess_image(image_path)  # Pré-processa a imagem
                features = extract_features(img_bin)  # Extrai as características
                feature_vectors.append(features)  # Armazena o vetor de características
                labels.append(class_mapping[class_name])  # Armazena o rótulo correspondente

    # Converte para numpy arrays
    return np.array(feature_vectors), np.array(labels)
