import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Connect to ChromaDB
# -----------------------------
db_client = chromadb.PersistentClient("./chroma_db")
collection = db_client.get_collection("notes")

# -----------------------------
# Load embedding model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Get user question
# -----------------------------
question = input("Ask: ")

# -----------------------------
# Generate embedding
# -----------------------------
query_embedding = embedding_model.encode(question)

# -----------------------------
# Retrieve relevant chunks
# -----------------------------
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
)

context = "\n\n".join(results["documents"][0])

# -----------------------------
# Prompt
# -----------------------------
prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}
"""

# -----------------------------
# OpenRouter Client
# -----------------------------
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# -----------------------------
# Call LLM
# -----------------------------
response = client.chat.completions.create(
    model="google/gemma-3-12b-it",
    messages=[
        {
            "role": "system",
            "content": "You answer questions only from the given context.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    temperature=0,
)

# -----------------------------
# Print Answer
# -----------------------------
print("\nAnswer:\n")
print(response.choices[0].message.content)