from fastapi import FastAPI
from pydantic import BaseModel
from app.services.processor import process_query
from app.rag.pipeline import run_rag

#FasAPI handles the HTTP request handling.

app=FastAPI()

@app.get("/health")
def health():
    return {"status":"Ok"}


class Queryvalidation(BaseModel):
    question:str

@app.post("/query")
def query(request:Queryvalidation):
    answer=run_rag(request.question)
    return {"response":answer}
