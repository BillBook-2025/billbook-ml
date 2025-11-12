import pinecone

class VectorDBService:
    def __init__(self, api_key: str, environment: str = "us-west1-gcp", index_name: str = "books-index", dimension: int = 384):
        """
        Pinecone 초기화 및 인덱스 연결
        """
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name

        # 인덱스 없으면 생성
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=dimension, metric="cosine")
        
        self.client = pinecone.Index(index_name)

    def add_vectors(self, ids: list[str], embeddings: list[list[float]], metadata: list[dict]):
        """벡터와 메타데이터를 DB에 저장"""
        vectors = [(i, e, m) for i, e, m in zip(ids, embeddings, metadata)]
        self.client.upsert(vectors=vectors)

    def query(self, embedding: list[float], top_k: int = 5):
        """쿼리 벡터로 DB에서 유사 벡터 검색"""
        return self.client.query(vector=embedding, top_k=top_k, include_metadata=True)