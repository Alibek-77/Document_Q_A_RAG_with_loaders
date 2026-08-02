from fastapi import UploadFile,File,HTTPException,status
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chroma_client import collection
import tempfile,os
async def upload_document(file:UploadFile=File(...)):
    allowed_files_types=["application/pdf","text/plain"]
    if file.content_type not in allowed_files_types:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="File type not supported")
    suffix=".pdf" if file.content_type=="application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
        content=await file.read()
        tmp.write(content)
        tmp_path=tmp.name
    try:
        if suffix=="pdf":
            loader=PyPDFLoader(file_path=tmp_path)
        else:
            loader=TextLoader(file_path=tmp_path)
        documents=loader.load()
        for doc in documents:
            doc.metadata["original_filename"]=file.filename
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n","\n","."," ",""]
        )
        chunks=splitter.split_documents(documents=documents)
        collection.add(
            ids=[f"chunk_{file.filename}_{i}" for i in range(len(chunks))],
            metadatas=[chunk.metadata for chunk in chunks],
            content=[chunk.page_content for chunk in chunks]
        )
        return {
            "message":f"File {file.filename} indexed",
            "chunks_count":len(chunks),
            "pages":len(documents)
        }
    finally:
        os.unlink(path=tmp_path)
    