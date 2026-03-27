from app.rag.ingest import load_documents, chunk_text, create_embeddings

text=load_documents("data/sample.txt")
print(f"Texts are {text}")

chunked_docs=chunk_text(text)
print(f"Chunks are {chunked_docs}")

embeddings=create_embeddings(chunked_docs)
print(f"Total length of embeddings are {embeddings}")