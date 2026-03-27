from sentence_transformers import SentenceTransformer


embedding_model=SentenceTransformer("all-MiniLM-L6-v2")

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
    embeddings=embedding_model.encode(chunks)
    return embeddings










