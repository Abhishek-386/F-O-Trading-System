# ============================================
# src/features/fo_specific.py - F&O Specific Features
# ============================================

import pandas as pd
import numpy as np


class FOFeatures:
    """Calculate F&O specific features"""
    
    def calculate_oi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Open Interest based features"""
        df = df.copy()
        
        # OI change
        df['oi_change'] = df['oi'].diff()
        df['oi_change_pct'] = df['oi'].pct_change() * 100
        
        # OI momentum
        df['oi_momentum_5'] = df['oi'].pct_change(5) * 100
        df['oi_momentum_10'] = df['oi'].pct_change(10) * 100
        
        # OI vs Volume ratio
        df['oi_volume_ratio'] = df['oi'] / df['volume'].rolling(window=20).mean()
        
        return df
    
    def calculate_pcr(
        self,
        option_chain: pd.DataFrame,
        by_volume: bool = False
    ) -> float:
        """
        Calculate Put-Call Ratio
        
        Args:
            option_chain: DataFrame with option chain data
            by_volume: If True, use volume instead of OI
            
        Returns:
            PCR value
        """
        metric = 'volume' if by_volume else 'oi'
        
        put_metric = option_chain[option_chain['type'] == 'PE'][metric].sum()
        call_metric = option_chain[option_chain['type'] == 'CE'][metric].sum()
        
        if call_metric == 0:
            return np.nan
        
        pcr = put_metric / call_metric
        return pcr
    
    def calculate_iv_skew(self, option_chain: pd.DataFrame) -> float:
        """Calculate IV skew (CE avg IV - PE avg IV)"""
        ce_iv = option_chain[option_chain['type'] == 'CE']['iv'].mean()
        pe_iv = option_chain[option_chain['type'] == 'PE']['iv'].mean()
        skew = ce_iv - pe_iv
        return skew
    
    def calculate_max_pain(self, option_chain: pd.DataFrame) -> float:
        """
        Calculate max pain strike
        
        Max pain is the strike where option writers (sellers) 
        would lose the least amount of money
        """
        strikes = option_chain['strike'].unique()
        pain_values = []
        
        for strike in strikes:
            # Calculate pain at this strike
            ce_data = option_chain[
                (option_chain['type'] == 'CE') & 
                (option_chain['strike'] <= strike)
            ]
            pe_data = option_chain[
                (option_chain['type'] == 'PE') & 
                (option_chain['strike'] >= strike)
            ]
            
            ce_pain = (ce_data['oi'] * (strike - ce_data['strike'])).sum()
            pe_pain = (pe_data['oi'] * (pe_data['strike'] - strike)).sum()
            
            total_pain = ce_pain + pe_pain
            pain_values.append((strike, total_pain))
        
        # Find strike with minimum pain
        max_pain_strike = min(pain_values, key=lambda x: x[1])[0]
        return max_pain_strike