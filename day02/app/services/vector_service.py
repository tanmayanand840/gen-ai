import chromadb

from app.config import CHROMA_DB_PATH
from app.utils.logger import logger
from app.exceptions import VectorDatabaseError


client = chromadb.PersistentClient(str(CHROMA_DB_PATH))

collection = client.get_collection("notes")


def search_documents(query_embedding, n_results=3):

    try:

        logger.info("Searching ChromaDB")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        documents = results["documents"][0]

        logger.info(
            f"Retrieved {len(documents)} documents"
        )

        return documents

    except Exception as e:

        logger.error(f"ChromaDB search failed: {e}")

        raise VectorDatabaseError(
            "Failed to search vector database"
        ) from e