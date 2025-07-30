import os, litserve as ls
from typing import Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
class RAGAPI(ls.LitAPI):
    def setup(self, device):
        self.emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vs = Chroma(persist_directory="chroma_db", embedding_function=self.emb)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    def predict(self, request: Dict[str, Any]):
        query = request.get("query", ""); k = int(request.get("top_k", 3))
        docs = self.vs.similarity_search(query, k=k)
        context = "\\n\\n".join(d.page_content for d in docs)
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
              {"role": "system", "content": "Answer using the provided context. If insufficient, say so."},
              {"role": "user", "content": f"Context:\\n{context}\\n\\nQuestion: {query}"}
            ],
            temperature=0.2,
        )
        return {"answer": resp.choices[0].message.content, "sources": [d.metadata for d in docs]}
if __name__ == "__main__":
    ls.LitServer(RAGAPI()).run(port=8001)
