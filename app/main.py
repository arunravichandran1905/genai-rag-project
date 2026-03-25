from fastapi import FastAPI

app=FastAPI()

@app.get("/health")
def health():
    return {"status": "Good"}

@app.post("/query")
def query():
    return {"response": "Query you send will be get processed and return to the server."}
