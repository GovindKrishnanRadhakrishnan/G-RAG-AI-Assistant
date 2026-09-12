from fastapi import FastAPI, Response, status
from fastapi.staticfiles import StaticFiles
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
import time
import requests
from src.api.routes import router, retriever
from src.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="G-RAG — local document Q&A with Ollama",
)

@app.get("/health")
async def health_check(response: Response):
    """
    Production-grade health check.
    Checks the status of the FastAPI app, local Ollama service, and Vector DB.
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "services": {
            "api": "up",
            "ollama": "unknown",
            "vector_db": "unknown"
        }
    }
    
    # Check Ollama status
    try:
        ollama_url = f"{settings.OLLAMA_HOST}/api/tags"
        r = requests.get(ollama_url, timeout=2)
        if r.status_code == 200:
            health_status["services"]["ollama"] = "healthy"
        else:
            health_status["services"]["ollama"] = f"unhealthy (status code: {r.status_code})"
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["services"]["ollama"] = f"unreachable ({str(e)})"
        health_status["status"] = "unhealthy"

    # Check Vector DB status (ChromaDB)
    try:
        if retriever and retriever.hybrid:
            health_status["services"]["vector_db"] = "healthy"
        else:
            health_status["services"]["vector_db"] = "uninitialized"
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["services"]["vector_db"] = f"error ({str(e)})"
        health_status["status"] = "unhealthy"

    if health_status["status"] == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
    return health_status

# Instrument and expose /metrics (only when prometheus_fastapi_instrumentator is installed)
if _PROMETHEUS_AVAILABLE:
    Instrumentator().instrument(app).expose(app)

app.include_router(router, prefix="/api/v1")

# Mount templates directory to serve index.html
app.mount("/", StaticFiles(directory="templates", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
