from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL
from app.utils.logger import logger
from app.exceptions import EmbeddingError

logger.info(
    f"Loading embedding model: {EMBEDDING_MODEL}"
)
model = SentenceTransformer(EMBEDDING_MODEL)


def create_embedding(text: str):
    try:
        logger.info("Creating query embedding")

        return model.encode(text).tolist()

    except Exception as e:

        logger.error(f"Embedding failed: {e}")

        raise EmbeddingError(
            "Failed to create embedding"
        ) from e