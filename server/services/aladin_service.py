import requests
from typing import List, Dict, Optional

class AladinBookFetchService:
    BASE_URL = "https://www.aladin.co.kr/ttb/api/ItemList.aspx"
    LOOKUP_URL = "https://www.aladin.co.kr/ttb/api/ItemLookUp.aspx"
    
    def __init__(self, ttb_key: str):
        self.ttb_key = ttb_key

    """
    알라딘 API를 호출해 인기책 / 베스트셀러 리스트를 가져옴
    
    :param query_type: API 호출 타입 (Bestseller, ItemNewAll, ItemNewSpecial, BlogBest 등)
    :param max_results: 가져올 책 개수 (1~100)
    :param category_id: 검색할 카테고리 ID. 0이면 전체. (예: 판타지 1101)
    :param search_target: 검색 대상 (Book, Music, DVD 등)
    :param output: 응답 포맷 (js, xml 등)
    :param version: API 버전
    :return: 책 정보 리스트
    """
    def fetch_books(
        self,
        query_type: str = "Bestseller",
        max_results: int = 10,
        search_target: str = "Book",
        category_id: int = 0,
        output: str = "js",
        version: str = "20131101"
    ) -> Optional[List[Dict]]:
        params = {
            "ttbkey": self.ttb_key,
            "QueryType": query_type,
            "MaxResults": max_results,
            "SearchTarget": search_target,
            "CategoryId": category_id,
            "output": output,
            "Version": version,
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            items = data.get("item", [])
            books = []
            
            for item in items:
                book = { # 기본 정보
                    "title": item.get("title"),
                    "author": item.get("author"),
                    "publisher": item.get("publisher"),
                    "cover": item.get("cover"),
                    "isbn13": item.get("isbn13"),
                }
                
                isbn13 = item.get("isbn13") # 상세 정보 추가
                if isbn13:
                    detail = self.fetch_book_detail(isbn13)
                    if detail:
                        book.update(detail)  # description, category 추가
                
                books.append(book)
                
            return books

        except requests.RequestException as e:
            print(f"❌ API 호출 오류: {e}")
            return None

    """
    알라딘 API를 통해 특정 도서의 상세 정보를 가져옴
    """
    def fetch_book_detail(self, isbn13: str):
        params = {
            "ttbkey": self.ttb_key,
            "itemIdType": "ISBN13",
            "ItemId": isbn13,
            "output": "js",
            "Version": "20131101",
            "Cover": "Big",
            "OptResult": "cateList,fulldescription,authors"
        }

        try:
            response = requests.get(self.LOOKUP_URL, params=params)
            response.raise_for_status()
            data = response.json()
    
            if "item" not in data or not data["item"]:
                return None
    
            item = data["item"][0]
            
            category_text = item.get("categoryName")
            category_parts = [c.strip() for c in category_text.split(">")][1:]

            return {
                "description": item.get("description", ""),
                "category": category_parts
            }
        
        except requests.RequestException as e:
            print(f"❌ 상세 조회 오류 ({isbn13}): {e}")
            return None