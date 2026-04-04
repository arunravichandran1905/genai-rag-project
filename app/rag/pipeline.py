from app.rag.ingest import load_documents, chunk_text, create_embeddings
from app.rag.retriever import VectorStore
from openai import OpenAI
import numpy as np
import os


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)


# global storage (temporary)
rag_data = None


def initialize_rag():
    global rag_data

    print("Initializing RAG...")

    text = load_documents("data/sample.txt")
    chunks = chunk_text(text)
    embeddings = create_embeddings(chunks)
    vector_store = VectorStore(embeddings)

    rag_data = {
        "chunks": chunks,
        "vector_store": vector_store
    }

    print("RAG initialized successfully")


def run_rag(query: str):
    global rag_data

    if rag_data is None:
        initialize_rag()

    client = get_client()

    chunks = rag_data["chunks"]
    vector_store = rag_data["vector_store"]

    # 🔹 Step 1: Create query embedding
    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    query_embedding = np.array([q_embedding]).astype("float32")

    # 🔹 Step 2: Retrieve relevant chunks
    index = vector_store.search(query_embedding, k=2)

    retrieved_chunks = [chunks[i] for i in index[0]]

    print("Retrieved chunks:", retrieved_chunks)

    context = "\n".join(retrieved_chunks)

    print("Context used:\n", context)

    # 🔹 Step 3: Prompt
    prompt = f"""
You must answer ONLY using the provided context.
If answer is not in context, say "I don't know".

Question: {query}
Context: {context}
"""

    # 🔹 Step 4: Call LLM (UPDATED MODEL)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print("FULL RESPONSE:", response)

    # 🔹 Step 5: Safe extraction
    content = response.choices[0].message.content

    if not content:
        print("⚠️ Empty response from model")
        return "Model returned empty response"

    return content