from services import AladinBookFetchService, VectorDBService
from ml_models import E5Embedding

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