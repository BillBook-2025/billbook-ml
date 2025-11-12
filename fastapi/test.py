from services import AladinBookFetchService, VectorDBService
from ml_models import e5_embedding

if __name__ == "__main__":
    TTB_KEY = "ttbseoll770145001"
    service = AladinBookFetchService(ttb_key=TTB_KEY)

    bestsellers = service.fetch_books(query_type="Bestseller", max_results=5)

    for idx, book in enumerate(bestsellers, start=1):    
        service = e5_embedding()
        result = service.embed_batch(book)