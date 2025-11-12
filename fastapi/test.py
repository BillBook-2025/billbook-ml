from services import AladinBookFetchService, VectorDBService
from ml_models import E5Embedding

"""
** 은 나중에 수정할거야 FT-Transformer(1)이나 SASRec(2)으로... 학습 잘 되면..

1. 알라딘 인기책 가져와서 전부다 e5로 임베딩하고, 벡터 db로 보내고
** 마지막으로 빌린 책을 기반으로 임베딩해서 책 6개 정도 추천하는걸로

2. 그리고 우리 DB에 있는 책 게시글에서 추천할건데,
** 사용자 seq를 임베딩하던 최근 5개만 요약하던지 해서.... 하기

결론! 주피터 써서 학습 계속 하는것도 좋은데...
일단 FastApi 써서 저런 로직들을 구성하자
"""

if __name__ == "__main__":
    TTB_KEY = "ttbseoll770145001"
    aladin_service = AladinBookFetchService(ttb_key=TTB_KEY)
    e5_service = E5Embedding()
    vertor_db_service = VectorDBService()

    # 책 불러오기
    bestsellers = aladin_service.fetch_books(query_type="Bestseller", max_results=50)

    # 임베딩 하기
    # 배치 텍스트와 book_id를 같은 순서로 저장하니 배치단위로 처리해도 매칭 안전함
    # batch_texts = ["book A text", "book B text", "book C text"]
    batch_texts = []
    book_ids = []
    for book in bestsellers:
        text = e5_service.build_text(book)
        batch_texts.append(text)
        book_ids.append(book['isbn13']) # 고유 ID, DB에 key로 사용
    embeddings = e5_service.embed_batch({"text": batch_texts})["embedding"]

    # 벡터DB에 넣기
    records = []
    for idx, emb in enumerate(embeddings):
        records.append({
            "id": book_ids[idx],
            "embedding": emb,
            "metadata": bestsellers[idx]  # 책 정보 그대로 넣기 가능
        })

    vertor_db_service.upsert_books(records)