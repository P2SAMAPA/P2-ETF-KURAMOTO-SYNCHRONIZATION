"""
trainer.py  —  Kuramoto ETF Synchronization Trainer
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from kuramoto_model import KuramotoETFAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_trainer() -> Dict:
    """Main Kuramoto trainer with multi-window support."""
    
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
        "window_results": {},
        "best_windows": {},
        "universes": {}
    }
    
    # Pass windows to analyzer
    config.KURAMOTO_CONFIG["WINDOWS"] = config.WINDOWS
    analyzer = KuramotoETFAnalyzer(config.KURAMOTO_CONFIG)
    
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {universe_name}...")
        
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue
        
        prices = prices_df[available].dropna().values
        returns = np.diff(np.log(prices), axis=0)
        
        if len(returns) < 100:
            logger.warning(f"Not enough data for {universe_name}")
            continue
        
        # Analyze universe with all windows
        result = analyzer.analyze_universe(returns, available)
        
        if "error" in result:
            logger.error(f"  Error: {result['error']}")
            continue
        
        results["top_picks"][universe_name] = result["picks"]
        results["sync_metrics"][universe_name] = result["sync_metrics"]
        results["window_results"][universe_name] = result.get("window_results", {})
        results["best_windows"][universe_name] = {
            "window": result.get("best_window", 252),
            "metrics": result.get("best_window_metrics", {})
        }
        results["universes"][universe_name] = {
            "tickers": available,
            "mode": result["mode"],
            "top_etfs": result["top_etfs"],
            "fit_results": result["fit_results"]
        }
        
        best_win = result.get("best_window", 252)
        logger.info(f"  ✅ Best window: {best_win} days")
        logger.info(f"  ✅ Mode: {result['mode']}")
        logger.info(f"  ✅ R_current: {result['sync_metrics']['R_current']:.3f}")
        logger.info(f"  ✅ Top picks:")
        for pick in result["picks"]:
            logger.info(f"     {pick['ticker']}: {pick['expected_return']}% ({pick['confidence']})")
    
    # Save results
    output_path = f"kuramoto_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Saved: {output_path}")
    
    # Upload to HuggingFace
    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    
    return results


if __name__ == "__main__":
    run_trainer()
