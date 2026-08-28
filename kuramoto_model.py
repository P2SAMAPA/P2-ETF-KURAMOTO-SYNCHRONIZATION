"""
kuramoto_model.py  —  Kuramoto Synchronization Model for ETFs
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class KuramotoETFModel:
    """
    Kuramoto model for ETF synchronization dynamics.
    
    Each ETF is an oscillator with:
    - phase theta_i(t)
    - natural frequency omega_i
    - coupling strength K
    
    The synchronization order parameter R(t) measures market coherence.
    """
    
    def __init__(self, n_oscillators: int, dt: float = 0.01, K: float = 1.0):
        self.n = n_oscillators
        self.dt = dt
        self.K = K
        
        # State variables
        self.theta = None
        self.omega = None
        self.A = None
        
        # History
        self.theta_history = []
        self.R_history = []
        self.psi_history = []
        
    def fit(self, returns: np.ndarray) -> Dict:
        """
        Fit Kuramoto model to ETF return data.
        """
        n_samples, n_oscillators = returns.shape
        
        # Estimate natural frequencies from mean returns
        self.omega = np.mean(returns, axis=0)
        
        # Estimate phases from cumulative returns
        self.theta = np.cumsum(returns, axis=0)
        self.theta = self.theta % (2 * np.pi)
        
        # Compute coupling matrix from correlations
        corr_matrix = np.corrcoef(returns.T)
        self.A = np.abs(corr_matrix)
        np.fill_diagonal(self.A, 0)
        
        # Find optimal coupling strength K
        self.K = self._estimate_coupling_strength(returns)
        
        # Compute synchronization order parameter
        R, psi = self._compute_order_parameter(self.theta[-1])
        
        return {
            "n_oscillators": n_oscillators,
            "omega": self.omega.tolist(),
            "K": self.K,
            "R": float(R),
            "psi": float(psi),
            "coupling_matrix": self.A.tolist()
        }
    
    def _estimate_coupling_strength(self, returns: np.ndarray) -> float:
        """Estimate optimal coupling strength from data."""
        n_samples = returns.shape[0]
        window = min(252, n_samples // 2)
        recent_returns = returns[-window:]
        
        R_ts = []
        for i in range(10, len(recent_returns)):
            theta = np.cumsum(recent_returns[:i], axis=0) % (2 * np.pi)
            R, _ = self._compute_order_parameter(theta[-1])
            R_ts.append(R)
        
        R_ts = np.array(R_ts)
        
        if np.std(R_ts) < 0.05:
            return 0.5
        
        R_derivative = np.diff(R_ts) / self.dt
        K_est = np.mean(R_derivative) / (1 - np.mean(R_ts))
        
        return np.clip(K_est, 0.1, 2.0)
    
    def _compute_order_parameter(self, theta: np.ndarray) -> Tuple[float, float]:
        """Compute Kuramoto order parameter R(t) and mean phase psi(t)."""
        complex_sum = np.mean(np.exp(1j * theta))
        R = np.abs(complex_sum)
        psi = np.angle(complex_sum)
        return R, psi
    
    def _kuramoto_derivative(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Kuramoto dynamics derivative."""
        dtheta = np.zeros_like(theta)
        
        for i in range(self.n):
            dtheta[i] = self.omega[i]
            for j in range(self.n):
                if i != j:
                    dtheta[i] += self.K * self.A[i, j] * np.sin(theta[j] - theta[i])
        
        return dtheta
    
    def integrate(self, theta0: np.ndarray, steps: int) -> np.ndarray:
        """Integrate Kuramoto dynamics forward in time."""
        t = np.linspace(0, steps * self.dt, steps)
        
        def ode_func(theta, t):
            return self._kuramoto_derivative(theta, t)
        
        result = odeint(ode_func, theta0, t)
        return result
    
    def predict_returns(self, current_theta: np.ndarray, horizon: int = 5) -> np.ndarray:
        """Predict future returns based on Kuramoto dynamics."""
        theta_future = self.integrate(current_theta, horizon)
        returns_future = np.diff(theta_future, axis=0)
        avg_returns = np.mean(returns_future, axis=0)
        return avg_returns
    
    def get_synchronization_metrics(self, returns: np.ndarray) -> Dict:
        """Compute synchronization metrics for ETF universe."""
        theta = np.cumsum(returns, axis=0) % (2 * np.pi)
        
        R_ts = []
        psi_ts = []
        
        for i in range(10, len(theta)):
            R, psi = self._compute_order_parameter(theta[i])
            R_ts.append(R)
            psi_ts.append(psi)
        
        R_ts = np.array(R_ts)
        psi_ts = np.array(psi_ts)
        
        R_mean = np.mean(R_ts)
        R_std = np.std(R_ts)
        R_trend = np.polyfit(range(len(R_ts)), R_ts, 1)[0]
        R_max = np.max(R_ts)
        R_min = np.min(R_ts)
        
        phase_velocity = np.diff(psi_ts) / self.dt
        freq_dispersion = np.std(phase_velocity)
        omega_std = np.std(self.omega) if self.omega is not None else 0
        
        return {
            "R_mean": float(R_mean),
            "R_std": float(R_std),
            "R_trend": float(R_trend),
            "R_max": float(R_max),
            "R_min": float(R_min),
            "R_current": float(R_ts[-1]) if len(R_ts) > 0 else 0,
            "freq_dispersion": float(freq_dispersion),
            "omega_std": float(omega_std),
            "sync_status": "SYNCHRONIZED" if R_mean > 0.7 else "DESYNCHRONIZED" if R_mean < 0.3 else "TRANSITIONING"
        }


class KuramotoETFAnalyzer:
    """Main analyzer for ETF synchronization trading with multi-window support."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.windows = config.get("WINDOWS", [126, 252, 504, 756, 1008])
        
    def backtest_window(self, returns: np.ndarray, tickers: List[str], window: int) -> Dict:
        """
        Backtest a specific window size.
        Uses rolling walk-forward validation.
        """
        n_samples = len(returns)
        
        if n_samples < window + 100:
            return {"error": "Insufficient data", "window": window}
        
        # Use 80% for training, 20% for testing
        train_size = int(n_samples * 0.8)
        
        predictions = []
        actuals = []
        R_history = []
        
        # Walk-forward
        for i in range(train_size, n_samples - 1):
            train_start = i - window
            if train_start < 0:
                continue
            
            returns_train = returns[train_start:i]
            returns_test = returns[i:i+1]
            
            try:
                model = KuramotoETFModel(
                    n_oscillators=returns.shape[1],
                    dt=self.config.get("dt", 0.01),
                    K=self.config.get("K", 1.0)
                )
                model.fit(returns_train)
                
                # Get sync metrics
                metrics = model.get_synchronization_metrics(returns_train)
                R_history.append(metrics["R_current"])
                
                # Predict next return
                current_theta = model.theta[-1] if model.theta is not None else None
                if current_theta is not None:
                    pred = model.predict_returns(current_theta, horizon=1)
                    predictions.append(pred)
                    actuals.append(returns_test[0])
            except:
                continue
        
        if len(predictions) < 10:
            return {"error": "Not enough predictions", "window": window}
        
        # Calculate performance metrics
        predictions = np.array(predictions).reshape(-1, returns.shape[1])
        actuals = np.array(actuals).reshape(-1, returns.shape[1])
        
        # Per-ETF performance
        etf_metrics = {}
        for i, ticker in enumerate(tickers):
            pred_i = predictions[:, i]
            actual_i = actuals[:, i]
            
            correlation = np.corrcoef(pred_i, actual_i)[0, 1] if len(pred_i) > 1 else 0
            mse = np.mean((pred_i - actual_i) ** 2)
            
            pred_sign = np.sign(pred_i)
            actual_sign = np.sign(actual_i)
            directional_acc = np.mean(pred_sign == actual_sign)
            
            returns_strategy = actual_i * pred_sign
            sharpe = np.mean(returns_strategy) / (np.std(returns_strategy) + 1e-8) * np.sqrt(252)
            
            etf_metrics[ticker] = {
                "correlation": float(correlation) if not np.isnan(correlation) else 0,
                "mse": float(mse),
                "directional_accuracy": float(directional_acc),
                "sharpe": float(sharpe)
            }
        
        # Overall metrics
        overall_correlation = np.mean([m["correlation"] for m in etf_metrics.values()])
        overall_sharpe = np.mean([m["sharpe"] for m in etf_metrics.values()])
        overall_directional = np.mean([m["directional_accuracy"] for m in etf_metrics.values()])
        
        return {
            "window": window,
            "n_predictions": len(predictions),
            "overall_correlation": float(overall_correlation),
            "overall_sharpe": float(overall_sharpe),
            "overall_directional": float(overall_directional),
            "etf_metrics": etf_metrics,
            "R_mean": float(np.mean(R_history)) if R_history else 0,
            "R_std": float(np.std(R_history)) if R_history else 0
        }
    
    def analyze_universe(self, returns: np.ndarray, tickers: List[str]) -> Dict:
        """Analyze a universe with multiple windows and find the best."""
        n_samples, n_etfs = returns.shape
        
        if n_samples < 50:
            return {"error": "Insufficient data"}
        
        # Backtest all windows
        window_results = {}
        for window in self.windows:
            if n_samples >= window + 100:
                logger.info(f"    Testing window {window}...")
                result = self.backtest_window(returns, tickers, window)
                if "error" not in result:
                    window_results[window] = result
                    logger.info(f"      Sharpe: {result['overall_sharpe']:.3f}, "
                               f"Directional: {result['overall_directional']:.2%}")
                else:
                    logger.warning(f"      {result['error']}")
        
        # Find best window by Sharpe ratio
        if window_results:
            best_window = max(window_results.items(), 
                            key=lambda x: x[1].get('overall_sharpe', -999))
            best_win = best_window[0]
            best_metrics = best_window[1]
        else:
            # Fallback to default
            best_win = 252
            best_metrics = {}
        
        # Train model on best window
        model = KuramotoETFModel(
            n_oscillators=n_etfs,
            dt=self.config.get("dt", 0.01),
            K=self.config.get("K", 1.0)
        )
        
        # Use last best_win days for training
        train_returns = returns[-best_win:] if len(returns) > best_win else returns
        fit_results = model.fit(train_returns)
        
        # Get synchronization metrics
        sync_metrics = model.get_synchronization_metrics(train_returns)
        
        # Predict returns
        current_theta = model.theta[-1] if model.theta is not None else None
        if current_theta is not None:
            predicted_returns = model.predict_returns(current_theta, horizon=5)
        else:
            predicted_returns = np.zeros(n_etfs)
        
        # Rank ETFs by predicted returns
        ranked_indices = np.argsort(predicted_returns)[::-1]
        
        # Determine mode
        is_dispersed = sync_metrics["R_mean"] < 0.5
        
        # Select top ETFs
        if is_dispersed:
            top_indices = ranked_indices[:self.config.get("TOP_N", 3)]
        else:
            phase_stability = []
            for i in range(n_etfs):
                theta_i = model.theta[-20:, i] if model.theta is not None else np.zeros(20)
                phase_stability.append(np.std(theta_i))
            stable_indices = np.argsort(phase_stability)[:self.config.get("TOP_N", 3)]
            top_indices = stable_indices
        
        # Build picks
        picks = []
        for idx in top_indices[:self.config.get("TOP_N", 3)]:
            ticker = tickers[idx]
            
            R = sync_metrics["R_current"]
            if is_dispersed:
                confidence = "High" if predicted_returns[idx] > np.mean(predicted_returns) else "Medium"
            else:
                confidence = "Medium" if R < 0.6 else "Low"
            
            picks.append({
                "ticker": ticker,
                "expected_return": round(predicted_returns[idx] * 100, 2),
                "confidence": confidence,
                "phase": float(current_theta[idx]) if current_theta is not None else 0,
                "phase_stability": float(np.std(model.theta[-20:, idx])) if model.theta is not None else 0
            })
        
        return {
            "picks": picks,
            "sync_metrics": sync_metrics,
            "fit_results": fit_results,
            "mode": "DISPERSION" if is_dispersed else "SYNCHRONIZATION",
            "top_etfs": [tickers[i] for i in ranked_indices[:5]],
            "best_window": best_win,
            "window_results": window_results,
            "best_window_metrics": best_metrics
        }
