from transformers import AutoTokenizer, AutoModel
import torch

class e5_embedding:
    def __init__(self):
        """
        HuggingFace 모델이랑 토크나이저 같은거 다 불러오자
        참고로 tokenizer은.. 문자를 모델의 입력으로 바꿔주는 도구래! 
        미리 학습되어있는거 불러오면 좋겠지?
        """
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small") # model_name 넣어주면 알아서 불러와줌
        self.model = AutoModel.from_pretrained("intfloat/e5-small").to(device)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 입력 텍스트 생성 (타이틀 + 설명 + 저자 등 결합)
    def build_text(self, row):
        parts = [
            f"Title: {row['title']}",
            f"Author: {row['authors']}",
            f"Publisher: {row['publisher']}",
            f"Category: {row['category']}",
            f"Description: {row['description']}"
        ]
        return " ".join( # 리스트의 문자열들을 공백으로 연결할건데.....
            [p for p in parts if isinstance(p, str)] # NaN이나 None이 있으면 제외함
        ) # 최종적으로 하나의 문장 형태로 반환한다고 함!! "Title: ... Author: ... Publisher: ... Description: ..."

    def embed_batch(self, batch):
        """
        embed_batch 함수:
        책 텍스트 리스트를 받아 E5 임베딩을 batch 단위로 만들어 반환함
        얘는 대량으로 처리하기 좋은 방법이라서..... 뭐 정기적으로 대량임베딩하거나 할때 좋대
        나중에 e5를 다른걸로 바꾸거나 뭐.... 모델 운영하다보면 semantic search가 부정확해지거나 할때 쓰면 굿

        어차피 우린 제목, 저자, 출판사, 설명 정도만 있어서... SBERT나 OpenAI emb 같은거 굳이...
        """
        self.model.eval()
        batch_texts = [f"passage: {t}" for t in batch["text"]] # 각 텍스트 앞에 passage:를 붙힘!(e5 권장; 문맥 signal)

        inputs = self.tokenizer(
            batch_texts, return_tensors="pt", 
            truncation=True, # 길이 넘치면 자름
            padding=True,  # batch 단위로 다른 문장 길이에 맞게 padding
            max_length=256 # 최대 토큰 길이 제한
        ).to(self.device)

        with torch.no_grad():  # gradient 비활성화
            outputs = self.model(**inputs) # 딕셔너리를 언패킹(**)하여 모델에 전달
        
            # 원래는 아웃풋이 다 토큰단위인데.... mean해줘서 문장단위로 임베딩하게 된다는뎅
            emb = outputs.last_hidden_state.mean(dim=1)  # mean pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1) # 정규화

        # pytorch 텐서는 기본적으로 연산 그래프를 추적해서 back-prop을 계산하나봐
        # 근데 .cpu().numpy()는 오직 gradient 추적 없는 순수 값(텐서)만 가능한 OP라서
        # .detach()를 통해서 그래프를 끊고 순수 값으로 탈바꿈 시킨대
        emb = emb.detach().cpu().numpy().tolist()
        return {"embedding": emb}