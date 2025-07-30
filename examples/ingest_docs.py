from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

DOCS_DIR = Path("docs"); DOCS_DIR.mkdir(exist_ok=True)
texts = []
for p in DOCS_DIR.rglob("*"):
    if p.suffix.lower() in {".txt", ".md"}:
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents(texts)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
Chroma.from_documents(chunks, emb, persist_directory="chroma_db").persist()
print(f"✅ Ingested {len(chunks)} chunks into ./chroma_db")
