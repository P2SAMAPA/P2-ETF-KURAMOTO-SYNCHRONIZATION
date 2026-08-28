"""
kuramoto_model.py  —  Kuramoto Model with Macro Variables
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class KuramotoETFModel:
    """
    Kuramoto model with macro variable integration.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.dt = config.get("dt", 0.01)
        self.K = config.get("K", 1.0)
        
    def compute_regime_phases(self, returns: np.ndarray, macro: np.ndarray) -> np.ndarray:
        """
        Compute phases based on volatility regimes AND macro variables.
        """
        n_samples, n_etfs = returns.shape
        
        # 1. Volatility component
        vol = np.abs(returns)
        vol_smooth = np.zeros_like(vol)
        for i in range(n_etfs):
            vol_smooth[:, i] = np.convolve(vol[:, i], np.ones(20)/20, mode='same')
        
        vol_norm = (vol_smooth - vol_smooth.mean(axis=0)) / (vol_smooth.std(axis=0) + 1e-8)
        
        # 2. Macro component
        macro_norm = (macro - macro.mean(axis=0)) / (macro.std(axis=0) + 1e-8)
        
        # 3. Combine: use macro to adjust phase
        # ETFs respond differently to macro variables
        phase = np.zeros((n_samples, n_etfs))
        
        for i in range(n_etfs):
            # Phase = volatility regime + macro influence
            # Each ETF has different sensitivity to macro
            macro_influence = np.mean(macro_norm, axis=1) * 0.3
            phase[:, i] = np.tanh(vol_norm[:, i] + macro_influence)
        
        return phase
    
    def compute_synchronization(self, phases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute synchronization parameter R(t).
        """
        n_samples, n_etfs = phases.shape
        
        R_ts = np.zeros(n_samples)
        psi_ts = np.zeros(n_samples)
        
        for t in range(n_samples):
            theta = phases[t] * np.pi
            complex_sum = np.mean(np.exp(1j * theta))
            R_ts[t] = np.abs(complex_sum)
            psi_ts[t] = np.angle(complex_sum)
        
        return R_ts, psi_ts
    
    def analyze_universe(self, returns: np.ndarray, macro: np.ndarray, 
                         tickers: List[str]) -> Dict:
        """
        Full analysis with macro variables.
        """
        n_samples, n_etfs = returns.shape
        
        if n_samples < 100:
            return {"error": "Insufficient data"}
        
        # Align macro to returns
        if len(macro) > n_samples:
            macro = macro[-n_samples:]
        elif len(macro) < n_samples:
            # Pad macro if needed
            pad = n_samples - len(macro)
            macro = np.vstack([macro[:1]] * pad + [macro])
        
        # Compute phases with macro
        phases = self.compute_regime_phases(returns, macro)
        
        # Compute synchronization
        R_ts, psi_ts = self.compute_synchronization(phases)
        
        # Current state
        R_current = R_ts[-1] if len(R_ts) > 0 else 0.5
        R_mean = np.mean(R_ts[-100:]) if len(R_ts) > 100 else np.mean(R_ts)
        R_std = np.std(R_ts)
        R_trend = np.polyfit(range(len(R_ts)), R_ts, 1)[0]
        
        # Determine mode
        is_dispersed = R_current < 0.5
        
        # Pick ETFs based on dispersion + macro
        if is_dispersed:
            # In dispersion mode, pick ETFs that benefit from macro conditions
            # Calculate macro alignment
            macro_latest = macro[-1] if len(macro) > 0 else np.zeros(1)
            
            # Each ETF's response to macro
            scores = []
            for i in range(n_etfs):
                # Correlation with macro
                corr = np.corrcoef(returns[:, i], macro.mean(axis=1))[0, 1]
                # Recent performance
                momentum = returns[-10:, i].mean()
                # Combine: favor high momentum + positive macro correlation
                score = momentum * 10 + corr * 0.5
                scores.append(score)
            
            ranked = sorted(range(n_etfs), key=lambda i: scores[i], reverse=True)
            top_indices = ranked[:self.config.get("TOP_N", 3)]
        else:
            # In synchronization mode, pick ETFs with stable macro exposure
            macro_corr = []
            for i in range(n_etfs):
                corr = np.corrcoef(returns[:, i], macro.mean(axis=1))[0, 1]
                macro_corr.append(abs(corr))
            
            # Pick ETFs with lowest macro correlation (most independent)
            ranked = sorted(range(n_etfs), key=lambda i: macro_corr[i])
            top_indices = ranked[:self.config.get("TOP_N", 3)]
        
        # Build picks
        picks = []
        for idx in top_indices:
            ticker = tickers[idx]
            expected_return = returns[-5:, idx].mean() * 100
            confidence = "High" if abs(expected_return) > 0.5 else "Medium" if abs(expected_return) > 0.2 else "Low"
            
            picks.append({
                "ticker": ticker,
                "expected_return": round(expected_return, 2),
                "confidence": confidence,
            })
        
        return {
            "picks": picks,
            "R_current": float(R_current),
            "R_mean": float(R_mean),
            "R_std": float(R_std),
            "R_trend": float(R_trend),
            "mode": "DISPERSION" if is_dispersed else "SYNCHRONIZATION",
            "sync_status": "SYNCHRONIZED" if R_current > 0.7 else "DESYNCHRONIZED" if R_current < 0.3 else "TRANSITIONING",
            "macro_used": True
        }
