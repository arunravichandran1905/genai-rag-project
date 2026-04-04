from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag.pipeline import run_rag

from openai import OpenAI

#FasAPI handles the HTTP request handling.

app=FastAPI()

@app.get("/health")
def health():
    return {"status":"Ok"}


class Queryvalidation(BaseModel):
    question:str

@app.post("/query")
def query(request:Queryvalidation):
    try:
        answer=run_rag(request.question)
        return {"response":answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    
