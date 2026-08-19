from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

from pydantic import BaseModel
from app.agent import answer_question

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QuestionRequest):
    return answer_question(req.question)