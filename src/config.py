# ============================================
# src/config.py - Configuration Settings
# ============================================

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_PATH: Path = BASE_DIR / "artifacts" / "models"
    
    # Database
    DATABASE_URL: str = "sqlite:///./fo_trading.db"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Trading Parameters
    LOT_SIZE: int = 25
    TRANSACTION_COST_BPS: float = 5.0
    SLIPPAGE_BPS: float = 5.0
    MARGIN_MULTIPLIER: float = 1.2
    
    # Risk Management
    MAX_POSITION_SIZE_PCT: float = 2.0
    MAX_DRAWDOWN_PCT: float = 15.0
    KELLY_FRACTION: float = 0.25
    
    # Model Settings
    DEFAULT_DIRECTION_MODEL: str = "lightgbm_direction_v1"
    DEFAULT_VOLATILITY_MODEL: str = "xgboost_volatility_v1"
    DEFAULT_REGIME_MODEL: str = "hmm_regime_v1"
    
    # Data Source
    DATA_SOURCE: str = "sample"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"


settings = Settings()