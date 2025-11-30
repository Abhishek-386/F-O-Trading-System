# ============================================
# src/api/routes/predict.py
# ============================================

from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
import numpy as np

from src.models.inference.predictor import DirectionPredictor, VolatilityPredictor
from src.features.technical import TechnicalFeatures
from src.features.fo_specific import FOFeatures

router = APIRouter()

direction_predictor = DirectionPredictor()
volatility_predictor = VolatilityPredictor()


class BarData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    oi: float


class OptionChainRow(BaseModel):
    timestamp: datetime
    strike: float
    expiry: str
    type: str  # CE or PE
    last_price: float
    bid: float
    ask: float
    oi: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float


class PredictionRequest(BaseModel):
    timestamp: datetime
    recent_bars: List[BarData]
    option_chain: List[OptionChainRow]


@router.post("/direction")
async def predict_direction(request: PredictionRequest):
    """
    Predict market direction for next period
    
    Returns probabilities for up/down/neutral and recommended action
    """
    try:
        # Convert to DataFrame for feature engineering
        import pandas as pd
        
        bars_df = pd.DataFrame([bar.dict() for bar in request.recent_bars])
        bars_df = bars_df.set_index('timestamp')
        
        # Calculate features
        tech_features = TechnicalFeatures()
        features_df = tech_features.calculate(bars_df)
        
        # Get latest row
        latest_features = features_df.iloc[-1:].copy()
        
        # Predict
        probs = direction_predictor.predict_proba(latest_features)
        label = direction_predictor.predict(latest_features)[0]
        
        # Determine recommended action
        prob_up, prob_down, prob_neutral = probs[0]
        
        if prob_up > 0.6:
            action = "BUY"
            confidence = prob_up
            strike_suggestion = _get_atm_strike(request.option_chain, offset=0)
            option_type = "CE"
        elif prob_down > 0.6:
            action = "SELL"
            confidence = prob_down
            strike_suggestion = _get_atm_strike(request.option_chain, offset=0)
            option_type = "PE"
        else:
            action = "NEUTRAL"
            confidence = prob_neutral
            strike_suggestion = None
            option_type = None
        
        # Calculate stop loss and target
        current_price = request.recent_bars[-1].close
        atr = features_df['atr_14'].iloc[-1]
        
        if action in ["BUY", "SELL"]:
            stop_loss = current_price - atr if action == "BUY" else current_price + atr
            target = current_price + 2 * atr if action == "BUY" else current_price - 2 * atr
        else:
            stop_loss = None
            target = None
        
        return {
            "timestamp": request.timestamp.isoformat(),
            "probabilities": {
                "up": float(prob_up),
                "down": float(prob_down),
                "neutral": float(prob_neutral),
            },
            "predicted_label": label,
            "recommended_action": action,
            "confidence": float(confidence),
            "option_suggestion": {
                "type": option_type,
                "strike": strike_suggestion,
                "stop_loss": float(stop_loss) if stop_loss else None,
                "target": float(target) if target else None,
            } if strike_suggestion else None,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/iv")
async def predict_iv(
    strike: float,
    expiry: str,
    recent_bars: List[BarData],
):
    """Predict implied volatility for given strike and expiry"""
    try:
        import pandas as pd
        
        # Convert bars to DataFrame
        bars_df = pd.DataFrame([bar.dict() for bar in recent_bars])
        bars_df = bars_df.set_index('timestamp')
        
        # Calculate features
        tech_features = TechnicalFeatures()
        features_df = tech_features.calculate(bars_df)
        
        # Add strike and expiry features
        latest_features = features_df.iloc[-1:].copy()
        latest_features['strike'] = strike
        latest_features['expiry'] = pd.to_datetime(expiry)
        latest_features['days_to_expiry'] = (
            latest_features['expiry'] - pd.Timestamp.now()
        ).dt.days
        
        # Predict IV
        iv_pred = volatility_predictor.predict(latest_features)[0]
        
        return {
            "strike": strike,
            "expiry": expiry,
            "predicted_iv": float(iv_pred),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_atm_strike(option_chain: List[OptionChainRow], offset: int = 0) -> float:
    """Get ATM or near ATM strike"""
    if not option_chain:
        return None
    
    # Get current spot price (approximate from option prices)
    strikes = [opt.strike for opt in option_chain]
    mid_strike = np.median(strikes)
    
    # Round to nearest strike
    strike_step = 50 if mid_strike > 20000 else 100
    atm_strike = round(mid_strike / strike_step) * strike_step
    
    return atm_strike + (offset * strike_step)