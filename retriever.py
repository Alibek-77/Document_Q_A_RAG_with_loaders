from schemas import AnswerResponse,QuestionResponse
from chroma_client import collection
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
load_dotenv()
model=ChatGoogleGenerativeAI(google_api_key=os.getenv("GEMINI_API_KEY"),model="gemini-2.5-flash")
def rrf_combine(bm25_docs: list, vector_docs: list, weight_bm25=0.5, weight_vector=0.5, k=60):
    scores = {}
    doc_map = {}
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0) + weight_bm25 * (1 / (k + rank + 1))
    for rank, doc in enumerate(vector_docs):
        doc_id = doc.page_content
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0) + weight_vector * (1 / (k + rank + 1))
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]
async def retriever(request:QuestionResponse):
    vector_results = collection.query(
        query_texts=[request.question],
        n_results=20
    )
    vector_docs = []
    if vector_results and "documents" in vector_results and vector_results["documents"]:
        for doc_text in vector_results["documents"][0]:
            vector_docs.append(Document(page_content=doc_text))
    all_chroma_docs = collection.get()
    if all_chroma_docs and "documents" in all_chroma_docs and all_chroma_docs["documents"]:
        all_documents = [Document(page_content=doc) for doc in all_chroma_docs["documents"]]
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = 5
        bm25_docs = bm25_retriever.invoke(request.question)
    else:
        bm25_docs = []
    hybrid_docs = rrf_combine(
        bm25_docs=bm25_docs,
        vector_docs=vector_docs,
        weight_bm25=0.5,
        weight_vector=0.5  
    )
    top_docs = hybrid_docs[:10]
    context_parts = [doc.page_content for doc in top_docs]
    context="\n\n---\n\n".join(context_parts)
    prompt=ChatPromptTemplate.from_messages([
        ("system","""Ты ассистент который отвечает строго на основе контекста.
        Перечисляй ВСЕ пункты и модули, упомянутые в контексте, ничего не пропуская.
Если информации нет в контексте — скажи "В предоставленных документах нет информации по этому вопросу."
Не выдумывай информацию.
Контекст из документов:
{context}"""),
            ("human","{question}")
    ])
    chain=prompt|model|StrOutputParser()
    answer= await chain.ainvoke({
        "question":request.question,
        "context":context
    })
    return AnswerResponse(answer=answer,sources=[]),context_parts