from app.rag.ingest import load_documents, chunk_text, create_embeddings
from app.rag.retriever import VectorStore
from openai import OpenAI
import numpy as np
import os

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text=load_documents("data/sample.txt")
chunks=chunk_text(text)
embeddings=create_embeddings(chunks)
vector_store=VectorStore(embeddings)


def run_rag(query:str):
    q_embedding=client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
    query_embedding=np.array([q_embedding]).astype("float32")

    index=vector_store.search(query_embedding, k=2)
    retrieved_chunks=[chunks[indices] for indices in index[0]]
    print(f"Retrieved chunks are: \n {retrieved_chunks}")
    context="\n".join(retrieved_chunks)

    prompts= f""" You must answer ONLY using the provided context.
If answer is not in context, say "I don't know".
    Question: {query}  
    Context: {context}
    """

    response=client.chat.completions.create(
        model="gpt-4", 
        messages=[
            {"role":"user", "content":prompts}
            ]
    )
    return response.choices[0].message.content






