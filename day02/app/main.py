from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.services.embedding_service import create_embedding
from app.services.vector_service import search_documents
from app.services.llm_service import generate_answer
from fastapi import HTTPException

from app.exceptions import (
    EmbeddingError,
    VectorDatabaseError,
    LLMError,
)


app = FastAPI(
    title="RAG API",
    description="AI-powered document question answering API",
    version="1.0.0",
)


# Request body
class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the documents"
    )


# Home
@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question

    try:

        # 1. Create embedding
        query_embedding = create_embedding(question)

        # 2. Retrieve relevant documents
        documents = search_documents(query_embedding)

        # 3. Combine retrieved chunks
        context = "\n\n".join(documents)

        # 4. Generate answer using LLM
        answer = generate_answer(
            question,
            context
        )

        # 5. Return response
        return {
            "question": question,
            "answer": answer
        }

    except EmbeddingError:
        raise HTTPException(
            status_code=500,
            detail="Could not create question embedding."
        )

    except VectorDatabaseError:
        raise HTTPException(
            status_code=500,
            detail="Could not search the document database."
        )

    except LLMError:
        raise HTTPException(
            status_code=503,
            detail="LLM service is currently unavailable."
        )