"""
streamlit_app.py  —  Kuramoto ETF Synchronization Dashboard
"""

import streamlit as st
import pandas as pd
import requests
import json
import glob
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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
    .sync-high { color: #e74c3c; font-weight: 700; }
    .sync-low { color: #27ae60; font-weight: 700; }
    .sync-medium { color: #f39c12; font-weight: 700; }
    .ticker-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
    }
    .best-window {
        background: #27ae60;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        display: inline-block;
    }
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


def create_sync_gauge(R_value: float):
    """Create a gauge chart for synchronization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=R_value,
        title={"text": "Synchronization (R)"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 1], "tickvals": [0, 0.3, 0.7, 1]},
            "bar": {"color": "#667eea"},
            "steps": [
                {"range": [0, 0.3], "color": "#27ae60"},
                {"range": [0.3, 0.7], "color": "#f39c12"},
                {"range": [0.7, 1], "color": "#e74c3c"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 0.7
            }
        }
    ))
    fig.update_layout(height=250)
    return fig


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
    tab1, tab2, tab3 = st.tabs(["📊 Top Picks", "🔄 Synchronization Metrics", "📈 Window Comparison"])
    
    top_picks = data.get('top_picks', {})
    sync_metrics = data.get('sync_metrics', {})
    window_results = data.get('window_results', {})
    best_windows = data.get('best_windows', {})
    
    with tab1:
        st.subheader("Top ETF Picks by Universe")
        st.markdown("*When markets are dispersed → relative-value opportunities*")
        
        for universe, picks in top_picks.items():
            metrics = sync_metrics.get(universe, {})
            R = metrics.get('R_current', 0.5)
            mode = "DISPERSED (Favorable)" if R < 0.5 else "SYNCHRONIZED (Cautious)"
            mode_color = "#27ae60" if R < 0.5 else "#e74c3c"
            
            best_win = best_windows.get(universe, {}).get('window', 252)
            
            st.markdown(f"### {universe}")
            st.markdown(f"**Best Window:** {best_win} days | **Mode:** <span style='color:{mode_color}'>{mode}</span> (R = {R:.3f})", unsafe_allow_html=True)
            
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
                        <div style="font-size:0.7rem; color:#888; margin-top:0.5rem;">
                            Phase Stability: {pick.get('phase_stability', 0):.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chart
            df = pd.DataFrame(picks)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['expected_return'],
                y=df['ticker'],
                orientation='h',
                text=df['expected_return'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside',
                marker_color=['#27ae60' if r > 0.5 else '#f39c12' if r > 0 else '#e74c3c' 
                              for r in df['expected_return']]
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"chart_picks_{universe}")
            st.markdown("---")
    
    with tab2:
        st.subheader("Synchronization Dynamics")
        
        for universe, metrics in sync_metrics.items():
            best_win = best_windows.get(universe, {}).get('window', 252)
            st.markdown(f"### {universe} (Best Window: {best_win} days)")
            
            col1, col2, col3, col4 = st.columns(4)
            
            R = metrics.get('R_current', 0)
            R_mean = metrics.get('R_mean', 0)
            R_trend = metrics.get('R_trend', 0)
            R_max = metrics.get('R_max', 0)
            
            with col1:
                st.metric("Current R", f"{R:.3f}", delta=f"{R - R_mean:.3f}")
            with col2:
                st.metric("Mean R", f"{R_mean:.3f}")
            with col3:
                st.metric("R Trend", f"{R_trend:.3f}", delta="↗" if R_trend > 0 else "↘")
            with col4:
                st.metric("Max R", f"{R_max:.3f}")
            
            fig_gauge = create_sync_gauge(R)
            st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{universe}")
            
            if R > 0.7:
                st.warning("⚠️ **High Synchronization** - Market is crowded. Risk of volatility expansion.")
            elif R < 0.3:
                st.success("✅ **Low Synchronization** - Markets are dispersed. Favorable for relative-value trading.")
            else:
                st.info("🔄 **Transitioning** - Markets are in transition. Watch for direction.")
            
            st.markdown("---")
    
    with tab3:
        st.subheader("Window Performance Comparison")
        st.markdown("*Compare Kuramoto model performance across different window sizes*")
        
        for universe, results in window_results.items():
            st.markdown(f"### {universe}")
            
            if not results:
                st.warning("No window results available")
                continue
            
            # Create dataframe
            df_windows = pd.DataFrame([
                {
                    "Window": w,
                    "Sharpe Ratio": r.get('overall_sharpe', 0),
                    "Directional Acc": r.get('overall_directional', 0) * 100,
                    "Correlation": r.get('overall_correlation', 0),
                    "Predictions": r.get('n_predictions', 0)
                }
                for w, r in results.items()
            ]).sort_values("Window")
            
            # Highlight best window
            best_idx = df_windows["Sharpe Ratio"].idxmax()
            
            # Display table
            st.dataframe(
                df_windows.style.apply(
                    lambda x: ['background-color: #90EE90' if x.name == best_idx else '' for i in x],
                    axis=1
                ),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Window": "Window (days)",
                    "Sharpe Ratio": st.column_config.NumberColumn("Sharpe", format="%.3f"),
                    "Directional Acc": st.column_config.NumberColumn("Directional", format="%.1f%%"),
                    "Correlation": st.column_config.NumberColumn("Correlation", format="%.3f"),
                    "Predictions": "N"
                }
            )
            
            # Chart: Sharpe vs Window
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_windows["Window"],
                y=df_windows["Sharpe Ratio"],
                mode='lines+markers',
                name='Sharpe Ratio',
                line=dict(color='blue', width=2),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=df_windows["Window"],
                y=df_windows["Directional Acc"],
                mode='lines+markers',
                name='Directional Accuracy %',
                line=dict(color='green', width=2),
                marker=dict(size=10),
                yaxis='y2'
            ))
            fig.update_layout(
                title="Window Performance Comparison",
                xaxis_title="Window Size (days)",
                yaxis_title="Sharpe Ratio",
                yaxis2=dict(
                    title="Directional Accuracy (%)",
                    overlaying='y',
                    side='right'
                ),
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_window_{universe}")
            
            # Best window recommendation
            best_win = int(df_windows.loc[best_idx, "Window"])
            best_sharpe = df_windows.loc[best_idx, "Sharpe Ratio"]
            best_dir = df_windows.loc[best_idx, "Directional Acc"]
            
            st.success(f"✅ **Recommended window: {best_win} days** "
                      f"(Sharpe: {best_sharpe:.3f}, Directional: {best_dir:.1f}%)")
            st.markdown("---")


if __name__ == "__main__":
    main()
