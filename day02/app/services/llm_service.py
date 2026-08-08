from openai import OpenAI

from app.config import OPENROUTER_API_KEY
from app.config import OPENROUTER_BASE_URL, LLM_MODEL
from app.utils.logger import logger
from app.exceptions import LLMError

logger.info("Sending request to LLM")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)



def generate_answer(question, context):
    prompt = f"""
You are a helpful AI assistant.

Answer only using the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
    )

    return response.output_text