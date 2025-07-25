from fastapi import FastAPI
from app.api.v1 import predict

app = FastAPI()

# router 등록
app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI!"}