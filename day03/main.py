import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

collection = client.create_collection("documents")

documents = [
    "Python is easy to learn.",
    "Artificial Intelligence is changing the world.",
    "Docker packages applications into containers.",
    "FastAPI is a modern Python framework."
]

embeddings = model.encode(documents)

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=["1", "2", "3", "4"]
)

query = "Tell me about AI."

query_embedding = model.encode(query)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=2
)

print(results["documents"])