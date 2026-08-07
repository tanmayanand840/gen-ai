from sentence_transformers import SentenceTransformer,util

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded!")

documents = [
    "Python is an easy programming language.",
    "Cricket is the most popular sport in India.",
    "The Earth revolves around the Sun.",
    "Machine Learning is a subset of Artificial Intelligence."
]

document_embeddings = model.encode(documents)

query = "Tell me about AI."

query_embedding = model.encode(query)


scores = util.cos_sim(query_embedding, document_embeddings)

best_match = scores.argmax()
print("Best Match:")
print(documents[best_match])
