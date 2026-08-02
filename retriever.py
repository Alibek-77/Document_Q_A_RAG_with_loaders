from schemas import AnswerResponse,QuestionResponse
from chroma_client import collection
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
model=ChatGoogleGenerativeAI(google_api_key=os.getenv("GEMINI_API_KEY"),model="gemini-2.5-flash")
async def retriever(request:QuestionResponse):
    results=collection.query(
        query_texts=request.question,
        n_results=request.top_k,
        include=["documents","metadatas","distances"]
    )
    if not results["documents"]:
        return AnswerResponse(
            answer="Did not found relevant information",
            sources=[]
        )
    sources=[]
    context_parts=[]
    for doc,meta,dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        context_parts.append(doc)
        sources.append({
            "filename":meta.get("original_filename","unknown"),
            "page":meta.get("page",0)+1,
            "relevance":round(1-dist,3)
        })
    context="\n\n---\n\n".join(context_parts)
    prompt=ChatPromptTemplate.from_messages([
        ("system","""Ты ассистент который отвечает строго на основе контекста.
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
    return AnswerResponse(answer=answer,sources=sources)