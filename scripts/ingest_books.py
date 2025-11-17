from server.services import VectorDBService, AladinBookFetchService
from server.ml_models import E5Embedding
import os
from dotenv import load_dotenv

"""
Pinecone에 데이터 넣는 초기 배치, 책 수집, E5 임베딩 생성 등
즉, 벡터 DB 적재용 스크립트
"""

load_dotenv()
ALADIN_KEY = os.getenv("ALADIN_KEY")
PINECONE_KEY = os.getenv("PINECONE_KEY")

aladin_service = AladinBookFetchService(ttb_key=ALADIN_KEY)
e5_service = E5Embedding()
pinecone_service = VectorDBService(api_key=PINECONE_KEY)

# 1. 책 불러오기
bestsellers = aladin_service.fetch_books(query_type="Bestseller", max_results=50)

# 2. 임베딩
batch_texts = [e5_service.build_text(book) for book in bestsellers]
book_ids = [book['isbn13'] for book in bestsellers]

embeddings = e5_service.embed_batch({"text": batch_texts})["embedding"]

# 3. 벡터 DB에 저장
pinecone_service.add_vectors(ids=book_ids, embs=embeddings, meta=bestsellers)

# 4. 테스트 쿼리
user_query = "드래곤이 나오는 판타지 소설"
test_emb = e5_service.embed_query(user_query)
results = pinecone_service.query(test_emb, top_k=5)['matches']

for r in results:
    print(r)