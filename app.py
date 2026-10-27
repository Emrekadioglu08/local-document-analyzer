from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from pypdf import PdfReader
import os
import shutil

app = FastAPI(
    title="Kurumsal RAG API",
    description="Yerel Ollama, ChromaDB ve FastAPI destekli güvenli AI PDF RAG Servisi",
    version="1.0"
)

# ChromaDB ve Embedding Model Başlatma
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_dokumanlari")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Kurumsal RAG Backend Servisi Çalışıyor 🚀"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları yüklenebilir.")
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        reader = PdfReader(temp_file_path)
        documents = []
        metadatas = []
        ids = []
        doc_counter = 0
        
        for sayfa_idx, sayfa in enumerate(reader.pages):
            metin = sayfa.extract_text()
            if metin:
                sayfa_no = sayfa_idx + 1
                chunk_size = 400
                overlap = 80
                for i in range(0, len(metin), chunk_size - overlap):
                    parca = metin[i:i + chunk_size]
                    if len(parca.strip()) > 20:
                        doc_counter += 1
                        documents.append(parca)
                        metadatas.append({"sayfa": sayfa_no, "dosya": file.filename})
                        ids.append(f"{file.filename}_p{sayfa_no}_{doc_counter}")
                        
        if documents:
            embeddings = embed_model.encode(documents).tolist()
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        
        os.remove(temp_file_path)
        return {
            "status": "success",
            "filename": file.filename,
            "total_chunks_indexed": len(documents)
        }
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Veritabanında henüz kayıtlı doküman bulunmuyor.")
        
    query_vector = embed_model.encode(request.question).tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2
    )
    
    if not results['documents'] or not results['documents'][0]:
        return {"answer": "İlgili bilgi veritabanında bulunamadı.", "source": None}
        
    best_text = results['documents'][0][0]
    meta = results['metadatas'][0][0]
    
    prompt = f"""
    Sen profesyonel bir kurumsal asistansın. Sana verilen kaynak metni kullanarak kullanıcının sorusunu Türkçe ve net yanıtla. Bilgi metinde yoksa "Bu bilgi metinde bulunmuyor" de.
    
    Kaynak Metin ({meta['dosya']} - Sayfa {meta['sayfa']}):
    {best_text}
    
    Soru: {request.question}
    """
    
    response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
    answer = response['message']['content']
    
    return {
        "answer": answer,
        "source": {
            "file": meta['dosya'],
            "page": meta['sayfa']
        }
    }

@app.get("/status")
async def get_status():
    return {"total_vectors_in_db": collection.count()}