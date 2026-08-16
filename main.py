"""
Healthcare RAG API
Local healthcare information retrieval and generation using FAISS, BGE embeddings, and Ollama.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Configuration
# =============================================================================

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
FAISS_DB_PATH = Path(os.getenv("FAISS_DB_PATH", "./faiss_rag_db"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# CPU / latency tuning for Ollama (adjust as needed)
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", "6"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "400"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "1h")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
MIN_TOP_K = 1
MAX_TOP_K = 10

# FAISS returns L2 distance by default (lower = more similar).
# Tune this if retrieval is too strict or too loose for your index.
RELEVANCE_DISTANCE_THRESHOLD = float(os.getenv("RELEVANCE_DISTANCE_THRESHOLD", "1.25"))

CONTENT_SNIPPET_MIN_CHARS = 300
CONTENT_SNIPPET_MAX_CHARS = 500

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

MISSING_INFO_ANSWER = (
    "I don't have enough relevant information in my medical database to answer that accurately. "
    "This assistant provides general educational information only and should not replace advice "
    "from a qualified healthcare professional."
)

HEALTHCARE_PROMPT_TEMPLATE = """
You are a knowledgeable, empathetic Healthcare Information Assistant.

Your job is to help users understand healthcare and medical information in a clear, natural, and easy-to-understand way.

Use the retrieved medical context as your primary factual knowledge source.

IMPORTANT RULES:

1. GROUNDEDNESS

Base factual medical claims primarily on the retrieved medical context.

Do not invent medical facts, statistics, treatments, diagnoses, or drug information that are not supported by the provided context.

If the context does not contain enough information to answer the question accurately, clearly say that the available medical database does not contain sufficient information.

Do not pretend that information exists in the database when it does not.

2. EXPLANATION

Do not simply copy the retrieved documents.

Synthesize and explain the relevant information naturally.

Use simple language where possible.

If a medical term is important, explain what it means.

The goal is understanding, not merely retrieving text.

3. CONVERSATIONAL BEHAVIOR

Respond naturally like a high-quality AI assistant.

Understand follow-up questions using the supplied chat history.

For example:

User:
"What is asthma?"

Follow-up:
"What are its symptoms?"

Understand that "its" refers to asthma.

Do not ask the user to repeat information that is already available in the conversation history.

4. RELEVANCE

Only use retrieved information that is relevant to the user's question.

Do not unnecessarily include unrelated medical information.

5. MISSING INFORMATION

If the retrieved context does not contain enough information to answer accurately, say:

"I don't have enough relevant information in my medical database to answer that accurately."

Do not hallucinate an answer.

If an external information source is supplied separately by the application, clearly label it as an external source rather than pretending it came from the internal database.

6. MEDICAL SAFETY

Do not diagnose the user.

Do not claim that the user has a particular disease or condition.

Do not provide personalized treatment plans.

Do not prescribe medications.

Do not provide specific medication dosages.

Do not tell the user to start, stop, or change prescription medication.

General educational information about medications and treatments is allowed when supported by the retrieved context.

For potentially urgent situations, provide general safety-oriented advice without pretending to perform a clinical assessment.

7. EMPATHY

Acknowledge the user's concern naturally when appropriate.

Do not use exaggerated empathy.

Avoid repetitive phrases such as:

"I'm sorry you're going through this."

unless they genuinely fit the situation.

8. DISCLAIMER

For medical-information responses, include a short, natural reminder that the information is educational and should not replace advice from a qualified healthcare professional.

Do not make the disclaimer unnecessarily long.

9. RESPONSE STYLE

Write like a modern conversational AI.

For simple questions:
- Answer directly.
- Keep the response relatively short.

For complex questions:
- Use short sections.
- Use bullet points when helpful.
- Bold important medical terms when appropriate.
- Explain concepts clearly.

Do not produce unnecessarily long responses.

10. CHAT HISTORY

Use the supplied chat history only to understand conversational context.

Do not treat previous assistant statements as authoritative medical evidence.

Medical factual claims should still be grounded in the retrieved medical context.

--------------------------

CHAT HISTORY:

{chat_history}

--------------------------

RETRIEVED MEDICAL CONTEXT:

{context}

--------------------------

USER QUESTION:

{question}

--------------------------

Provide the most helpful, accurate, clear, and natural answer possible.
"""

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("healthcare_rag_api")

# =============================================================================
# Pydantic Schemas
# =============================================================================


class QueryRequest(BaseModel):
    question: str = Field(..., description="User's healthcare question")
    chat_history: str = Field(default="", description="Optional prior conversation context")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only.")
        return cleaned

    @field_validator("chat_history")
    @classmethod
    def normalize_chat_history(cls, value: str) -> str:
        return value.strip()


class SourceDocument(BaseModel):
    disease: str
    source: str
    link: str = "N/A"
    content_snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    latency_seconds: float
    sources: list[SourceDocument]
    external_sources: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    service: str
    embedding_model: str
    llm_model: str
    vector_store: str
    vector_store_path: str
    components_loaded: dict[str, bool]


# =============================================================================
# Startup helpers
# =============================================================================


def _validate_faiss_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS vector database not found at '{path.resolve()}'. "
            "Ensure the index exists at ./faiss_rag_db before starting the API."
        )
    if not path.is_dir():
        raise FileNotFoundError(
            f"FAISS path '{path.resolve()}' exists but is not a directory."
        )


def _load_embeddings() -> HuggingFaceEmbeddings:
    try:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # Warm up once to fail fast if the model cannot be downloaded/loaded.
        _ = embeddings.embed_query("startup health check")
        logger.info("Embedding model loaded successfully.")
        return embeddings
    except Exception as exc:
        raise RuntimeError(
            "Failed to load embedding model "
            f"'{EMBEDDING_MODEL_NAME}'. "
            "Install dependencies with: pip install sentence-transformers torch "
            "and ensure you have internet access for the first model download."
        ) from exc


def _load_faiss_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    _validate_faiss_path(FAISS_DB_PATH)
    try:
        logger.info("Loading FAISS index from: %s", FAISS_DB_PATH.resolve())
        vector_store = FAISS.load_local(
            folder_path=str(FAISS_DB_PATH),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS vector store loaded successfully.")
        return vector_store
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FAISS index from '{FAISS_DB_PATH.resolve()}'. "
            "Verify the index was built with the same embedding model "
            f"({EMBEDDING_MODEL_NAME}) and that the folder is complete."
        ) from exc


def _check_ollama_model_available(model_name: str) -> None:
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10.0)
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama, then retry. "
            "On Windows, open the Ollama app or run: ollama serve"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Failed to query Ollama at {OLLAMA_BASE_URL}. "
            "Ensure Ollama is running."
        ) from exc

    payload = response.json()
    available_models = {
        item.get("name", "")
        for item in payload.get("models", [])
        if isinstance(item, dict)
    }

    normalized_target = model_name.strip()
    model_found = any(
        name == normalized_target or name.startswith(f"{normalized_target}:")
        for name in available_models
    )

    if not model_found:
        raise RuntimeError(
            f"Ollama model '{model_name}' is not installed. "
            f"Run: ollama pull {model_name}"
        )


def _create_ollama_llm() -> OllamaLLM:
    _check_ollama_model_available(OLLAMA_MODEL)
    try:
        logger.info("Initializing Ollama LLM: %s", OLLAMA_MODEL)
        llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=OLLAMA_TEMPERATURE,
            num_thread=OLLAMA_NUM_THREAD,
            num_ctx=OLLAMA_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
        # Quick inference check to fail fast at startup.
        _ = llm.invoke("Reply with exactly: OK")
        logger.info("Ollama LLM initialized successfully.")
        return llm
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Ollama model '{OLLAMA_MODEL}'. "
            f"Ensure Ollama is running and the model is pulled: ollama pull {OLLAMA_MODEL}"
        ) from exc


# =============================================================================
# RAG helpers
# =============================================================================


def _make_content_snippet(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= CONTENT_SNIPPET_MAX_CHARS:
        return cleaned

    snippet = cleaned[:CONTENT_SNIPPET_MAX_CHARS]
    if len(cleaned) > CONTENT_SNIPPET_MIN_CHARS:
        last_space = snippet.rfind(" ")
        if last_space >= CONTENT_SNIPPET_MIN_CHARS:
            snippet = snippet[:last_space]

    return snippet.rstrip() + "..."


def _extract_source_document(document: Any, score: float) -> SourceDocument:
    metadata = document.metadata if hasattr(document, "metadata") else {}
    if not isinstance(metadata, dict):
        metadata = {}

    page_content = getattr(document, "page_content", "") or ""
    link = metadata.get("link") or metadata.get("url") or "N/A"

    return SourceDocument(
        disease=str(metadata.get("disease", "Unknown")),
        source=str(metadata.get("source", "Unknown")),
        link=str(link) if link else "N/A",
        content_snippet=_make_content_snippet(page_content),
    )


def _retrieve_relevant_documents(
    vector_store: FAISS,
    question: str,
    top_k: int,
) -> list[tuple[Any, float]]:
    """
    Retrieve documents with similarity scores.
    LangChain FAISS typically returns L2 distance (lower = more similar).
    """
    results = vector_store.similarity_search_with_score(question, k=top_k)
    relevant: list[tuple[Any, float]] = []

    for document, score in results:
        if score <= RELEVANCE_DISTANCE_THRESHOLD:
            relevant.append((document, float(score)))

    logger.info(
        "Retrieved %d/%d documents within relevance threshold (<= %.2f).",
        len(relevant),
        len(results),
        RELEVANCE_DISTANCE_THRESHOLD,
    )
    return relevant


def _build_context_block(relevant_docs: list[tuple[Any, float]]) -> str:
    if not relevant_docs:
        return "No relevant medical context was retrieved from the internal database."

    sections: list[str] = []
    for index, (document, score) in enumerate(relevant_docs, start=1):
        metadata = document.metadata if hasattr(document, "metadata") else {}
        if not isinstance(metadata, dict):
            metadata = {}

        disease = metadata.get("disease", "Unknown")
        source = metadata.get("source", "Unknown")
        link = metadata.get("link") or metadata.get("url") or "N/A"
        content = getattr(document, "page_content", "") or ""

        sections.append(
            f"[Internal Medical Database - Chunk {index}]\n"
            f"Disease: {disease}\n"
            f"Source: {source}\n"
            f"Link: {link}\n"
            f"Relevance Distance: {score:.4f}\n"
            f"Content:\n{content}"
        )

    return "\n\n".join(sections)


def _build_prompt(
    question: str,
    chat_history: str,
    context: str,
    external_sources: list[str] | None = None,
) -> str:
    external_block = ""
    if external_sources:
        joined_sources = "\n".join(f"- {source}" for source in external_sources)
        external_block = (
            "\n\n--------------------------\n\n"
            "EXTERNAL WEB SOURCES (NOT from internal medical database):\n\n"
            f"{joined_sources}\n"
        )

    prompt = HEALTHCARE_PROMPT_TEMPLATE.format(
        chat_history=chat_history if chat_history else "No prior conversation.",
        context=context,
        question=question,
    )
    return prompt + external_block


def _process_query_sync(
    vector_store: FAISS,
    llm: OllamaLLM,
    question: str,
    chat_history: str,
    top_k: int,
    external_sources: list[str] | None = None,
) -> tuple[str, list[SourceDocument]]:
    relevant_docs = _retrieve_relevant_documents(vector_store, question, top_k)

    if not relevant_docs:
        return MISSING_INFO_ANSWER, []

    context = _build_context_block(relevant_docs)
    prompt = _build_prompt(
        question=question,
        chat_history=chat_history,
        context=context,
        external_sources=external_sources,
    )

    answer = llm.invoke(prompt)
    if not isinstance(answer, str):
        answer = str(answer)

    sources = [_extract_source_document(doc, score) for doc, score in relevant_docs]
    return answer.strip(), sources


# =============================================================================
# FastAPI application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Healthcare RAG API...")

    try:
        embeddings = _load_embeddings()
        vector_store = _load_faiss_store(embeddings)
        llm = _create_ollama_llm()

        app.state.embeddings = embeddings
        app.state.vector_store = vector_store
        app.state.llm = llm
        app.state.startup_ok = True

        logger.info("All components loaded successfully.")
    except Exception as exc:
        app.state.startup_ok = False
        app.state.startup_error = str(exc)
        logger.exception("Application startup failed: %s", exc)
        raise

    yield

    logger.info("Shutting down Healthcare RAG API.")


app = FastAPI(
    title="Healthcare RAG API",
    description=(
        "Local healthcare information retrieval and generation API "
        "using FAISS, BGE embeddings, and Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    state = request.app.state
    components_loaded = {
        "embeddings": getattr(state, "embeddings", None) is not None,
        "vector_store": getattr(state, "vector_store", None) is not None,
        "llm": getattr(state, "llm", None) is not None,
    }

    return HealthResponse(
        status="online" if all(components_loaded.values()) else "degraded",
        service="Healthcare RAG API",
        embedding_model=EMBEDDING_MODEL_NAME,
        llm_model=OLLAMA_MODEL,
        vector_store="FAISS",
        vector_store_path=str(FAISS_DB_PATH),
        components_loaded=components_loaded,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest, request: Request) -> QueryResponse:
    if not getattr(request.app.state, "startup_ok", False):
        raise HTTPException(
            status_code=500,
            detail="Service is not fully initialized. Check server logs.",
        )

    vector_store: FAISS = request.app.state.vector_store
    llm: OllamaLLM = request.app.state.llm

    question = payload.question
    logger.info("Received query (top_k=%d).", payload.top_k)

    start_time = time.perf_counter()

    try:
        # CPU-bound work off the event loop
        answer, sources = await asyncio.to_thread(
            _process_query_sync,
            vector_store,
            llm,
            question,
            payload.chat_history,
            payload.top_k,
            [],  # external_sources placeholder for future web-search integration
        )
    except Exception as exc:
        logger.exception("RAG query processing failed.")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the healthcare query. Please try again later.",
        ) from exc

    latency_seconds = round(time.perf_counter() - start_time, 2)
    logger.info("Query completed in %.2f seconds.", latency_seconds)

    return QueryResponse(
        question=question,
        answer=answer,
        latency_seconds=latency_seconds,
        sources=sources,
        external_sources=[],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )