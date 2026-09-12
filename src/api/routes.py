<<<<<<< HEAD
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..config import get_settings
from ..rag.document_processor import DocumentProcessor
from ..rag.retriever import AdvancedRetriever

router = APIRouter()
settings = get_settings()

# Initialize Advanced RAG singleton
retriever = AdvancedRetriever(
    persist_directory=settings.VECTOR_DB_PATH,
    ollama_host=settings.OLLAMA_HOST,
    ollama_model=settings.OLLAMA_MODEL,
    top_k_retrieval=settings.TOP_K_RETRIEVAL,
    top_k_final=settings.TOP_K_FINAL,
    compression_threshold=settings.COMPRESSION_THRESHOLD,
    memory_turns=settings.MEMORY_TURNS,
    cross_encoder_model=settings.CROSS_ENCODER_MODEL,
    embed_model_name=settings.EMBED_MODEL,
    run_evaluation=settings.RUN_EVALUATION,
)
doc_processor = DocumentProcessor(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)


class ChatRequest(BaseModel):
    question: str


@router.post("/upload-paper")
async def upload_paper(file: UploadFile = File(...)):
    """Upload a PDF, extract chunks, and ingest into the vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    try:
        content = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = doc_processor.process_pdf(tmp_path)
            for chunk in chunks:
                chunk["metadata"]["source_file"] = file.filename

            retriever.ingest(chunks)

            return {
                "message": f"Processed {file.filename}",
                "chunks_added": len(chunks),
                "filename": file.filename,
                "status": "ready",
            }
        finally:
            os.unlink(tmp_path)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "We couldn't process this document. "
                "Check that the file is a valid PDF and try again."
            ),
        )


@router.post("/chat")
async def chat(request: ChatRequest):
    """Run the document RAG pipeline for a user question."""
    try:
        result = retriever.query(request.question)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "faithfulness": result.get("evaluation", {}).get("faithfulness", 0.0),
            "relevance": result.get("evaluation", {}).get("relevance", 0.0),
            "rewritten_query": result.get("rewritten_query"),
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="We couldn't answer that question. Please try again.",
        )


@router.post("/clear")
async def clear_conversation():
    """Clear conversation memory used by the RAG pipeline."""
    retriever.clear_memory()
    return {"message": "Conversation cleared", "status": "ok"}
=======
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..config import get_settings
from ..rag.document_processor import DocumentProcessor
from ..rag.retriever import AdvancedRetriever

router = APIRouter()
settings = get_settings()

# Initialize Advanced RAG singleton
retriever = AdvancedRetriever(
    persist_directory=settings.VECTOR_DB_PATH,
    ollama_host=settings.OLLAMA_HOST,
    ollama_model=settings.OLLAMA_MODEL,
    top_k_retrieval=settings.TOP_K_RETRIEVAL,
    top_k_final=settings.TOP_K_FINAL,
    compression_threshold=settings.COMPRESSION_THRESHOLD,
    memory_turns=settings.MEMORY_TURNS,
    cross_encoder_model=settings.CROSS_ENCODER_MODEL,
    embed_model_name=settings.EMBED_MODEL,
    run_evaluation=settings.RUN_EVALUATION,
    enable_query_rewrite=settings.ENABLE_QUERY_REWRITE,
    enable_rerank=settings.ENABLE_RERANK,
    enable_compression=settings.ENABLE_COMPRESSION,
    llm_num_predict=settings.LLM_NUM_PREDICT,
    ollama_keep_alive=settings.OLLAMA_KEEP_ALIVE,
)
doc_processor = DocumentProcessor(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
)


class ChatRequest(BaseModel):
    question: str


class ResetDocumentsRequest(BaseModel):
    pin: str


@router.post("/upload-paper")
async def upload_paper(file: UploadFile = File(...)):
    """Upload a PDF, extract chunks, and ingest into the vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    try:
        content = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = doc_processor.process_pdf(tmp_path)
            for chunk in chunks:
                chunk["metadata"]["source_file"] = file.filename

            retriever.ingest(chunks)

            return {
                "message": f"Processed {file.filename}",
                "chunks_added": len(chunks),
                "filename": file.filename,
                "status": "ready",
            }
        finally:
            os.unlink(tmp_path)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "We couldn't process this document. "
                "Check that the file is a valid PDF and try again."
            ),
        )


@router.post("/chat")
async def chat(request: ChatRequest):
    """Run the document RAG pipeline for a user question."""
    try:
        result = retriever.query(request.question)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "faithfulness": result.get("evaluation", {}).get("faithfulness"),
            "relevance": result.get("evaluation", {}).get("relevance"),
            "rewritten_query": result.get("rewritten_query"),
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="We couldn't answer that question. Please try again.",
        )


@router.post("/clear")
async def clear_conversation():
    """Clear conversation memory used by the RAG pipeline."""
    retriever.clear_memory()
    return {"message": "Conversation cleared", "status": "ok"}


@router.post("/reset-documents")
async def reset_documents(request: ResetDocumentsRequest):
    """Delete all uploaded documents after verifying the safety PIN."""
    if request.pin != settings.RESET_DOCUMENTS_PIN:
        raise HTTPException(
            status_code=403,
            detail="Incorrect PIN. Documents were not deleted.",
        )
    try:
        retriever.reset_documents()
        return {
            "message": "All documents have been deleted.",
            "status": "ok",
            "chunks_remaining": retriever.vector_store.count(),
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="We couldn't reset the documents. Please try again.",
        )
>>>>>>> 0975fe4 (feat: speed up RAG pipeline and add PIN-protected document reset)
