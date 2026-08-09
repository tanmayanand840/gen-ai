from app.services.embedding_service import create_embedding
from app.services.vector_service import search_documents
from app.services.llm_service import generate_answer
from app.exceptions import (
    EmbeddingError,
    VectorDatabaseError,
    LLMError,
)


question = input("Ask: ")


try:

    query_embedding = create_embedding(question)

    documents = search_documents(query_embedding)

    context = "\n\n".join(documents)

    answer = generate_answer(question, context)

    print("\nAnswer:\n")
    print(answer)

except EmbeddingError:
    print("Could not process your question.")

except VectorDatabaseError:
    print("Could not search the documents.")

except LLMError:
    print("Could not generate an answer.")