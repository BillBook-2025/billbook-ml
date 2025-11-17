from pydantic import BaseModel
from typing import List

class BookMetadata(BaseModel): # 결과 모델 정의
    isbn13: str
    title: str
    cover: str
    author: str
    publisher: str
    category: list[str]
    description: str
    score: float

class BookSearchResponse(BaseModel):
    results: List[BookMetadata]