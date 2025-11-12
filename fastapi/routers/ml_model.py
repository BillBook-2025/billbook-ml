# 실제로 예측 담당

# ml_model.py
import joblib
from typing import List

# 이미 학습된 모델 파일을 로드
model = joblib.load("models/trained_model.pkl")

def predict(features: List[float]) -> float:
    # features를 모델에 넣어 예측 결과 리턴
    prediction = model.predict([features])
    return prediction[0]


# ml_model.py
import torch
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np

# 학습된 모델을 불러오거나 생성
# (이 예제에선 미리 학습된 모델을 저장해놓고 불러온다고 가정)


MODEL_PATH = "knn_model.joblib"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def predict_with_knn(model, input_data):
    # input_data는 예: {'features': [5.1, 3.5, 1.4, 0.2]}
    features = np.array(input_data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]