from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from config import CHROMA_COLLECTION, CHROMA_DB_PATH, EMBEDDING_MODEL, PROJECT_DIR

with open(PROJECT_DIR / "data" / "notes.txt", "r", encoding="utf-8") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

model = SentenceTransformer(EMBEDDING_MODEL)

embeddings = model.encode(chunks)

client = chromadb.PersistentClient(str(CHROMA_DB_PATH))

collection = client.get_or_create_collection(CHROMA_COLLECTION)

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist(),
)
