from book_fetch_service import AladinBookFetchService

if __name__ == "__main__":
    TTB_KEY = "ttbexample1234567890"
    service = AladinBookFetchService(ttb_key=TTB_KEY)

    bestsellers = service.fetch_books(query_type="Bestseller", max_results=5)

    for idx, book in enumerate(bestsellers, start=1):
        print(f"{idx}. {book['title']} - {book['author']} ({book['publisher']})")