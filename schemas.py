from pydantic import BaseModel
class QuestionResponse(BaseModel):
    question:str
    top_k:int=3
class AnswerResponse(BaseModel):
    answer:str
    sources:list[dict]