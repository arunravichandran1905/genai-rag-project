from fastapi import FastAPI
from pydantic import BaseModel
from app.services.processor import process_query

app=FastAPI()

@app.get("/health")
def health():
    return {"status":"Ok"}


class Queryvalidation(BaseModel):
    question:str

@app.post("/query")
def query(request:Queryvalidation):
    question=request.question
    formatted_question=process_query(question)
    return {"response":formatted_question}

