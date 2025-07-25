from fastapi import FastAPI, APIRouter
from pydantic import BaseModel


# BaseModel 상속받으면 JSON 데이터 이런거 쉽게 받을 수 있음
class PredictInput(BaseModel):
    data: dict



@router.post("/predict")
def predict(data: dict):
    # 여기에 PyTorch 모델 예측 코드 들어감
    return {"result": "dummy result"}


router = APIRouter()
knn_model = load_model()

@router.post("/predict")
def predict(data: dict):
    prediction = predict_with_knn(knn_model, data)
    return {"result": str(prediction)}
