from pinecone import Pinecone, ServerlessSpec

class VectorDBService:
    def __init__(self, api_key: str, 
                 index_name: str = "books-index", 
                 dimension: int = 384):
        """
        이 클래스로 말하자면,,, pinecone을 편하게 쓰기 위한 wrapper임!!
        Pinecone 초기화 및 인덱스 연결
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # 인덱스 없으면 생성
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        # 인덱스 연결
        self.client = self.pc.Index(index_name)

    def reset_index(self):
        """ DB에 리셋 """
        self.client.delete(delete_all=True)
        print("Pinecone index has been reset!")
    
    def add_vectors(self, ids: list[str], embeddings: list[list[float]], metadata: list[dict]):
        """ 벡터와 메타데이터를 DB에 저장 """
        vectors = [
            {
                "id": i,
                "values": e,
                "metadata": m
            }
            for i, e, m in zip(ids, embeddings, metadata)
        ]
        
        self.client.upsert(vectors=vectors)

    def query(self, embedding: list[float], top_k: int = 5):
        """ 쿼리 벡터로 DB에서 유사 벡터 검색 """
        return self.client.query(vector=embedding, top_k=top_k, include_metadata=True)