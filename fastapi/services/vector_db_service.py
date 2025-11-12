class VectorDBService:
    def __init__(self, client):
        self.client = client  # FAISS, Pinecone 등 DB 클라이언트

    def add_vectors(self, ids: list[str], embeddings: list[list[float]], metadata: list[dict]):
        """벡터와 메타데이터를 DB에 저장"""
        self.client.upsert(vectors=zip(ids, embeddings, metadata))

    def query(self, embedding: list[float], top_k: int = 5):
        """쿼리 벡터로 DB에서 유사 벡터 검색"""
        return self.client.query(vector=embedding, top_k=top_k)