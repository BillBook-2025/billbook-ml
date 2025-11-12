import requests
from typing import List, Dict, Optional

class AladinBookFetchService:
    BASE_URL = "https://www.aladin.co.kr/ttb/api/ItemList.aspx"

    def __init__(self, ttb_key: str):
        self.ttb_key = ttb_key

    def fetch_books(
        self,
        query_type: str = "Bestseller",
        max_results: int = 100,
        search_target: str = "Book",
        output: str = "js",
        version: str = "20131101"
    ) -> Optional[List[Dict]]:
        """
        알라딘 API를 호출해 인기책 / 베스트셀러 리스트를 가져옵니다.

        :param query_type: API 호출 타입 (Bestseller, ItemNewAll, ItemNewSpecial, BlogBest 등)
        :param max_results: 가져올 책 개수 (1~100)
        :param search_target: 검색 대상 (Book, Music, DVD 등)
        :param output: 응답 포맷 (js, xml 등)
        :param version: API 버전
        :return: 책 정보 리스트
        """
        params = {
            "ttbkey": self.ttb_key,
            "QueryType": query_type,
            "MaxResults": max_results,
            "SearchTarget": search_target,
            "output": output,
            "Version": version,
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # 결과 파싱
            items = data.get("item", [])
            books = [
                {
                    "title": item.get("title"),
                    "author": item.get("author"),
                    "publisher": item.get("publisher"),
                    "price": item.get("priceStandard"),
                    "cover": item.get("cover"),
                    "link": item.get("link")
                }
                for item in items
            ]
            return books

        except requests.RequestException as e:
            print(f"❌ API 호출 오류: {e}")
            return None