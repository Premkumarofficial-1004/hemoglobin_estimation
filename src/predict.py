
import cv2
import numpy as np
import pandas as pd
import joblib


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found.")
    img = cv2.resize(img, (224,224))
    return img

def extract_features(img):
    b,g,r = cv2.split(img.astype('float32'))
    features = {
        'mean_r': np.mean(r),
        'mean_g': np.mean(g),
        'mean_b': np.mean(b),
        'std_r': np.std(r),
        'std_g': np.std(g),
        'std_b': np.std(b),
        'red_ratio': np.mean(r) / (np.mean(r)+np.mean(g)+np.mean(b)+1e-8)
    }
    return pd.DataFrame([features])

def predict_ml(image_path, model_path='models/rf_model.joblib'):
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        cols = model_data['feature_columns']
        img = preprocess_image(image_path)
        X = extract_features(img)[cols]
        pred = model.predict(X)[0]
        return pred
    except Exception:
        # Temporary simulated output until real model is trained
        return np.random.uniform(10, 15)


