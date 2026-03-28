from app.rag.ingest import load_documents, chunk_text, create_embeddings
from app.rag.retriever import VectorStore
from sentence_transformers import SentenceTransformer

text=load_documents("data/sample.txt")
print(f"Texts are {text}")

chunked_docs=chunk_text(text)
print(f"Chunks are {chunked_docs}")

embeddings=create_embeddings(chunked_docs)
print(f"Total length of embeddings are {len(embeddings)} and {embeddings}")

vector_store=VectorStore(embeddings)

embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
query="What is AI ?"
q_embeddings=embedding_model.encode([query])

indices=vector_store.search(q_embeddings, k=2)

for index in indices[0]:
    print(chunked_docs[index])





