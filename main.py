from fastapi import FastAPI,UploadFile,File
from indexer import upload_document
from retriever import retriever
from schemas import QuestionResponse
app=FastAPI(title="Document Q&A",version="1.0.1")
@app.post("/documents/upload")
async def upload_documents(file:UploadFile=File(...)):
    return await upload_document(file=file)
@app.post("/ask")
async def ask_question(request:QuestionResponse):
    return await retriever(request=request)
@app.get("/health")
def health():
    return{"status":"ok"}