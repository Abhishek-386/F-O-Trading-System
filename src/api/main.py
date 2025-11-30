# ============================================
# src/api/main.py - FastAPI Application Entry
# ============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from src.api.routes import health, models, predict, backtest, data
from src.config import settings
from src.database.session import init_db

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting F&O Trading System API")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="F&O Trading System API",
    description="Production ML-powered Futures & Options trading system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(backtest.router, prefix="/backtest", tags=["Backtesting"])
app.include_router(data.router, prefix="/data", tags=["Data"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
