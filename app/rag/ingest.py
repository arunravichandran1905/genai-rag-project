from openai import OpenAI
import numpy as np

client=OpenAI()
def load_documents(path:str):
    with open(path, "r") as f:
        text=f.read()
    return text
    

def chunk_text(text:str, chunk_size:int=100):
    chunks=[]
    for i in range(0, len(text), chunk_size):
        chunk=text[i:i+chunk_size]
        chunks.append(chunk)
        print(f"{chunks} from chunking")
    return chunks


def create_embeddings(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    embeddings = [e.embedding for e in response.data]

    return np.array(embeddings).astype("float32")









