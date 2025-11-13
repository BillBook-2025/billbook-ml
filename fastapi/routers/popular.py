from fastapi import APIRouter
from services import VectorDBService
from ml_models import E5Embedding
from models import BookSearchResponse
from config import PINECONE_KEY

router = APIRouter()
pinecone_service = VectorDBService(api_key=PINECONE_KEY)
e5_service = E5Embedding()

@router.get("/search", response_model=BookSearchResponse)
async def search_books(query: str):
  index = pinecone_service.Index("books-index")
  query_vector = e5_service.embed_batch({"text": query})[0]["embedding"]
  res = index.query(query_vector, top_k = 5, include_metadata=True)

  results = []
  for match in res['matches']:
    metadata = match.get('metadata', {})
    results.append({
      "isbn13": metadata.get('isbn13', ''),
      "title": metadata.get('title', ''),
      "cover": metadata.get('cover', ''),
      "author": metadata.get('author', ''),
      "publisher": metadata.get('publisher', ''),
      "category": metadata.get('category', []),
      "description": metadata.get('description', ''),
      "score": match.get('score', 0)
  })
  
  return {"results": results}