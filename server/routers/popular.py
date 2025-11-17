from fastapi import APIRouter
from server.services import VectorDBService
from server.ml_models import E5Embedding
from server.models import BookSearchResponse
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_KEY = os.getenv("PINECONE_KEY")

router = APIRouter()
pinecone_service = VectorDBService(api_key=PINECONE_KEY)
e5_service = E5Embedding()

@router.get("/search", response_model=BookSearchResponse)
async def search_books(query: str):
  index = pinecone_service.client
  query_vector = e5_service.embed_query(query)
  res = index.query(
    vector=query_vector, 
    top_k=5, 
    include_metadata=True)

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