import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

# LlamaIndex Imports
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

app = FastAPI(title="Local Document Analyzer API", version="2.0")

# 1. LLM ve Embedding Ayarları (Ollama & HuggingFace)
print("Initializing LLM and Embedding models...")
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Global Değişkenler
UPLOAD_DIR = "./uploaded_docs"
STORAGE_DIR = "./llama_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)

index = None

# Başlangıçta daha önceden kaydedilmiş bir index varsa yükle
if os.path.exists(STORAGE_DIR) and os.listdir(STORAGE_DIR):
    try:
        print("Loading existing index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded successfully!")
    except Exception as e:
        print(f"Could not load index: {e}")

class QueryRequest(BaseModel):
    question: str

@app.get("/status")
def get_status():
    global index
    return {
        "status": "connected",
        "index_loaded": index is not None,
        "model": "llama3.2",
        "embedding": "all-MiniLM-L6-v2"
    }

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    global index
    try:
        # Dosyaları yükleme klasörüne kaydet
        saved_files = []
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_files.append(file_path)

        # LlamaIndex ile dokümanları oku ve indeksle
        print(f"Reading documents from {UPLOAD_DIR}...")
        documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
        
        print("Creating VectorStoreIndex...")
        index = VectorStoreIndex.from_documents(documents)
        
        # İndeksi diske kaydet
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        
        return {
            "message": "Files uploaded and indexed successfully",
            "files": [f.filename for f in files],
            "total_documents": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_documents(request: QueryRequest):
    global index
    if not index:
        raise HTTPException(status_code=400, detail="No document index found. Please upload documents first.")
    
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(request.question)
        return {
            "response": str(response),
            "sources": [node.node.metadata.get("file_name", "Unknown") for node in response.source_nodes]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary")
def get_document_summary():
    global index
    if not index:
        return {"error": "Henüz yüklenmiş bir doküman bulunamadı."}
    
    try:
        query_engine = index.as_query_engine(response_mode="tree_summarize")
        prompt = (
            "Bu belgenin/dokümanların kapsamlı bir yönetici özetini (Executive Summary), "
            "ana başlıklarını içeren bir içindekiler tablosunu (Outline) ve "
            "en kritik 5 ana vurguyu (Key Highlights) Türkçe olarak hazırla."
        )
        response = query_engine.query(prompt)
        return {"summary": str(response)}
    except Exception as e:
        return {"error": f"Özet çıkarılırken bir hata oluştu: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)