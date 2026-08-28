"""
streamlit_app.py  —  Kuramoto ETF Synchronization Dashboard
"""

import streamlit as st
import pandas as pd
import requests
import json
import glob
from datetime import datetime

st.set_page_config(
    page_title="P2-KURAMOTO-SYNC",
    page_icon="🔄",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .ticker-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
    }
    .confidence-high { color: #27ae60; font-weight: 600; }
    .confidence-medium { color: #f39c12; font-weight: 600; }
    .confidence-low { color: #e74c3c; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)


def load_data():
    """Load latest results."""
    json_files = glob.glob("kuramoto_results_*.json")
    if json_files:
        latest = sorted(json_files)[-1]
        with open(latest, 'r') as f:
            return json.load(f)
    
    try:
        repo_id = "P2SAMAPA/p2-etf-kuramoto-sync-results"
        today = datetime.now().strftime("%Y-%m-%d")
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/kuramoto_results_{today}.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    return None


def main():
    st.markdown('<div class="main-header">🔄 P2-KURAMOTO-SYNC</div>', unsafe_allow_html=True)
    st.markdown("*ETF Synchronization Dynamics from Statistical Physics*")
    
    data = load_data()
    
    if not data:
        st.error("No data available. Run `python trainer.py` first.")
        return
    
    run_date = data.get('run_date', 'Unknown')
    st.caption(f"Results from: {run_date}")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Top Picks", "📈 Window Performance"])
    
    top_picks = data.get('top_picks', {})
    sync_metrics = data.get('sync_metrics', {})
    best_windows = data.get('best_windows', {})
    
    with tab1:
        st.subheader("Top ETF Picks by Universe")
        
        for universe, picks in top_picks.items():
            metrics = sync_metrics.get(universe, {})
            R = metrics.get('R_current', 0.5)
            best_win = best_windows.get(universe, {}).get('window', 252)
            mode = metrics.get('mode', 'UNKNOWN')
            
            st.markdown(f"### {universe}")
            st.markdown(f"**Best Window:** {best_win} days | **R(t):** {R:.3f} | **Mode:** {mode}")
            
            if R > 0.7:
                st.warning("⚠️ High Synchronization - Market is crowded")
            elif R < 0.3:
                st.success("✅ Low Synchronization - Favorable for selection")
            else:
                st.info("🔄 Transitioning - Watch for direction")
            
            cols = st.columns(min(len(picks), 3))
            for i, pick in enumerate(picks):
                with cols[i % len(cols)]:
                    conf = pick['confidence'].lower()
                    color = "#27ae60" if conf == "high" else "#f39c12" if conf == "medium" else "#e74c3c"
                    
                    st.markdown(f"""
                    <div class="ticker-card">
                        <h3 style="margin:0;">{pick['ticker']}</h3>
                        <div style="font-size:2rem; font-weight:700; margin:0.5rem 0;">
                            {pick['expected_return']:.1f}%
                        </div>
                        <div style="color:{color}; font-weight:600;">Confidence: {pick['confidence']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
    
    with tab2:
        st.subheader("Window Performance - Top 3 ETFs by Universe")
        
        for universe in sync_metrics.keys():
            st.markdown(f"### {universe}")
            
            universe_data = sync_metrics.get(universe, {})
            window_results = universe_data.get('window_results', {})
            
            if not window_results:
                st.warning(f"No window results available for {universe}")
                continue
            
            # Display each window's picks
            for window, result in sorted(window_results.items()):
                window_int = int(window)
                picks = result.get('picks', [])
                R_current = result.get('R_current', 0)
                mode = result.get('mode', 'UNKNOWN')
                
                if picks:
                    ticker_str = ', '.join([f"{p['ticker']} ({p['expected_return']:.1f}%)" for p in picks])
                    st.markdown(f"**{window_int}d** (R={R_current:.3f}, {mode}): {ticker_str}")
                else:
                    st.markdown(f"**{window_int}d**: No picks")
            
            st.markdown("---")


if __name__ == "__main__":
    main()
