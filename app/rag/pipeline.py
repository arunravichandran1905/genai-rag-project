from sentence_transformers import SentenceTransformer
from app.rag.ingest import load_documents, chunk_text, create_embeddings, embedding_model
from app.rag.retriever import VectorStore
from openai import OpenAI

client=OpenAI()

text=load_documents("data/sample.txt")
chunks=chunk_text(text)
embeddings=create_embeddings(chunks)
vector_store=VectorStore(embeddings)


def run_rag(query:str):
    q_embedding=embedding_model.encode([query])
    index=vector_store.search(q_embedding, k=2)

    retrieved_chunks=[chunks[indices] for indices in index[0]]
    print(f"Retrieved chunks are: \n {retrieved_chunks}")

    prompts= f""" Answer the question by given Context
    Question: {query}  
    Context: {retrieved_chunks}
    """

    response=client.chat.completions.create(
        model="gpt-4", 
        messages=[
            {"role":"user", "content":prompts}
            ]
    )
    return response.choices[0].message.content






