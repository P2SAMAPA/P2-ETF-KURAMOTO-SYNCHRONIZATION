"""
trainer.py  —  Kuramoto Trainer with Macro Variables
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from kuramoto_model import KuramotoETFModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_trainer() -> Dict:
    """Main trainer with macro variables."""
    
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}
    
    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date,
        "top_picks": {},
        "sync_metrics": {},
        "best_windows": {},
        "universes": {},
        "macro_used": True
    }
    
    model = KuramotoETFModel(config.KURAMOTO_CONFIG)
    
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name}...")
        
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue
        
        prices = prices_df[available].dropna().values
        returns = np.diff(np.log(prices), axis=0)
        
        # Get macro variables
        macro = macro_df.values
        
        # Align macro with returns
        if len(macro) > len(returns):
            macro = macro[-len(returns):]
        elif len(macro) < len(returns):
            pad = len(returns) - len(macro)
            macro = np.vstack([macro[:1]] * pad + [macro])
        
        if len(returns) < 100:
            logger.warning(f"Not enough data for {universe_name}")
            continue
        
        # Test different windows
        window_results = {}
        best_score = -999
        best_window = 252
        
        for window in config.WINDOWS:
            if len(returns) < window + 50:
                continue
            
            returns_window = returns[-window:]
            macro_window = macro[-window:]
            
            result = model.analyze_universe(returns_window, macro_window, available)
            
            if "error" not in result:
                window_results[window] = {
                    "R_current": result["R_current"],
                    "R_mean": result["R_mean"],
                    "mode": result["mode"],
                    "picks": result["picks"]
                }
                
                # Score: dispersion is good (R < 0.5)
                score = 1.0 - result["R_mean"]
                if score > best_score:
                    best_score = score
                    best_window = window
        
        # Use best window
        if best_window in window_results:
            final_result = window_results[best_window]
            picks = final_result["picks"]
            mode = final_result["mode"]
            R = final_result["R_current"]
        else:
            # Fallback
            fallback = model.analyze_universe(returns[-252:], macro[-252:], available)
            picks = fallback.get("picks", [])
            mode = fallback.get("mode", "UNKNOWN")
            R = fallback.get("R_current", 0.5)
        
        results["top_picks"][universe_name] = picks
        results["sync_metrics"][universe_name] = {
            "R_current": R,
            "mode": mode,
            "window_results": window_results
        }
        results["best_windows"][universe_name] = {"window": best_window}
        results["universes"][universe_name] = {
            "tickers": available,
            "macro_cols": list(macro_df.columns)
        }
        
        logger.info(f"  ✅ Best window: {best_window}")
        logger.info(f"  ✅ Mode: {mode} (R={R:.3f})")
        logger.info(f"  ✅ Macro variables used: {list(macro_df.columns)}")
        logger.info(f"  ✅ Top picks:")
        for pick in picks:
            logger.info(f"     {pick['ticker']}: {pick['expected_return']}% ({pick['confidence']})")
    
    # Save results
    output_path = f"kuramoto_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Saved: {output_path}")
    
    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    
    return results


if __name__ == "__main__":
    run_trainer()
