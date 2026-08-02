from pydantic import BaseModel
class QuestionResponse(BaseModel):
    question:str
    top_k:int=3
class AnswerResponse(BaseModel):
    anaswer:str
    sources:list[dict]