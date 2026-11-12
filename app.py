from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.data_loader import read_uploaded_csv
from engine.indicators import add_indicators
from engine.market_structure import add_market_structure
from engine.outcomes import label_forward_returns, outcome_summary
from engine.patterns import detect_patterns
from providers.fivepaisa import connection_status

st.set_page_config(
    page_title="Chart Reader | Machine Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Chart Reader — Machine Intelligence")
st.caption("An evidence-led market analysis laboratory · 5paisa integration in progress")

provider = connection_status()
with st.sidebar:
    st.header("Project controls")
    st.info("Research build · Read-only · No order placement")
    st.markdown("**Broker:** 5paisa")
    st.markdown("**Deployment:** Streamlit Community Cloud")
    st.divider()
    st.subheader("Data source")
    uploaded = st.file_uploader("Upload historical OHLCV CSV", type=["csv"])
    st.caption("Required columns: timestamp (or date), open, high, low, close. Volume is optional.")
    symbol = st.text_input("Instrument label", value="Uploaded instrument")
    st.caption("Uploaded data stays in this app session; it is not written to GitHub.")

st.subheader("Workspace")
overview, chart_tab, intelligence, library_tab, research = st.tabs(
    ["Overview", "Chart analysis", "Intelligence", "Pattern library", "Research & validation"]
)

data = None
error = None
if uploaded is not None:
    try:
        raw = read_uploaded_csv(uploaded)
        data = add_market_structure(detect_patterns(add_indicators(raw)))
        data = label_forward_returns(data, horizons=(3, 6, 12))
    except ValueError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"Unexpected data-processing error: {exc}"

with overview:
    a, b, c = st.columns(3)
    a.metric("5paisa connection", "Pending")
    b.metric("Analysis engine", "Available")
    c.metric("Trading mode", "Read-only")
    st.warning(provider.message)
    if error:
        st.error(error)
    elif data is not None:
        st.success(f"Loaded and validated {len(data):,} candles from your CSV.")
        x, y, z = st.columns(3)
        x.metric("Latest close", f"{data['close'].iloc[-1]:,.2f}")
        y.metric("Detected pattern candles", f"{(data['detected_patterns'] != '').sum():,}")
        z.metric("Data rows", f"{len(data):,}")
    else:
        st.info("Upload a historical OHLCV CSV in the sidebar to use the independent analysis engine while 5paisa access is being clarified.")

    st.markdown("### Development roadmap")
    st.markdown(
        """
        - [x] Public Streamlit application and secure secret-storage pattern
        - [x] CSV OHLCV ingestion and validation
        - [x] Initial indicators and explainable candle detectors
        - [x] Trend / swing structure and forward-return research labels
        - [ ] Automated unit-test execution and broader edge-case coverage
        - [ ] 5paisa authentication and verified historical-data adapter
        - [ ] Live read-only market monitoring
        - [ ] Walk-forward ML experiments and paper validation
        """
    )

with chart_tab:
    st.markdown("### Price chart")
    if data is None:
        st.info("Upload a CSV to render its candles and indicators.")
    else:
        view = data.tail(500).copy()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=view["timestamp"], open=view["open"], high=view["high"],
            low=view["low"], close=view["close"], name="OHLC"
        ))
        for column, label in [("ema_20", "EMA 20"), ("ema_50", "EMA 50")]:
            fig.add_trace(go.Scatter(
                x=view["timestamp"], y=view[column], mode="lines",
                name=label, connectgaps=False
            ))
        fig.update_layout(
            height=650, xaxis_rangeslider_visible=False,
            template="plotly_dark", legend_orientation="h",
            margin=dict(l=10, r=10, t=35, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing latest {len(view)} candles. Indicators use only current and past observations.")

with intelligence:
    st.markdown("### Explainable candle intelligence")
    if data is None:
        st.info("Upload a CSV to inspect detected candle patterns and market context.")
    else:
        latest = data.iloc[-1]
        st.subheader(f"{symbol}: latest candle")
        p, q, r = st.columns(3)
        p.metric("Trend state", str(latest["trend_state"]))
        q.metric("RSI (14)", "—" if pd.isna(latest["rsi_14"]) else f"{latest['rsi_14']:.1f}")
        r.metric("ATR (14)", "—" if pd.isna(latest["atr_14"]) else f"{latest['atr_14']:.3f}")
        patterns = data.loc[data["detected_patterns"] != "", ["timestamp", "close", "detected_patterns"]].tail(30)
        st.markdown("#### Recent detected patterns")
        if patterns.empty:
            st.write("No registered patterns detected in this sample.")
        else:
            st.dataframe(patterns, use_container_width=True, hide_index=True)
        st.caption("Patterns are descriptive candle shapes, not standalone buy/sell signals.")


with library_tab:
    st.markdown("### Candle & pattern behaviour research registry")
    registry_path = Path(__file__).parent / "data" / "pattern_registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        entries = registry["patterns"]
        families = sorted({item["family"] for item in entries})
        selected_family = st.selectbox("Pattern family", ["All"] + families)
        visible = entries if selected_family == "All" else [
            item for item in entries if item["family"] == selected_family
        ]
        st.caption(
            f"{len(visible)} of {len(entries)} catalogue entries · "
            f"{sum(item['detector_status'] == 'implemented' for item in visible)} "
            "currently have detector code"
        )
        labels = {f"{item['name']} ({item['id']})": item for item in visible}
        selected_label = st.selectbox("Inspect pattern", list(labels))
        item = labels[selected_label]
        st.markdown(f"**Conventional classification:** {item['bias']}")
        st.markdown(f"**Definition:** {item['definition']}")
        st.markdown(f"**Behaviour hypothesis:** {item['hypothesis']}")
        st.markdown(f"**Context and confirmation:** {item['context']}")
        st.markdown(f"**Failure / invalidation:** {item['failure']}")
        st.info("Research caveat: " + item["caveat"])
        st.caption(
            "Detector status: " + item["detector_status"]
            + " · Empirical performance: " + item["empirical_performance"]
        )
        st.markdown("#### Research references")
        for ref in registry["references"]:
            st.markdown(f"- [{ref['author']} — {ref['title']}]({ref['url']})")
    except Exception as exc:
        st.error(f"Could not load pattern registry: {exc}")

with research:
    st.markdown("### Historical outcome research")
    if data is None:
        st.info("Upload a CSV to calculate forward returns for detected patterns.")
    else:
        horizon = st.selectbox("Forward horizon (candles)", [3, 6, 12], index=0)
        pattern_map = {
            "Doji": "pattern_doji",
            "Hammer": "pattern_hammer",
            "Shooting star": "pattern_shooting_star",
            "Bullish engulfing": "pattern_bullish_engulfing",
            "Bearish engulfing": "pattern_bearish_engulfing",
            "Inside bar": "pattern_inside_bar",
        }
        rows = []
        for label, column in pattern_map.items():
            result = outcome_summary(data, column, horizon)
            rows.append({
                "Pattern": label,
                "Observations": result["observations"],
                "Mean forward return (%)": result["mean_return_pct"],
                "Positive outcomes (%)": result["positive_rate_pct"],
            })
        summary = pd.DataFrame(rows)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "Forward returns are measured from the pattern candle close to the selected future close. "
            "They exclude transaction costs, slippage and execution constraints; they are not strategy backtest results."
        )
        st.download_button(
            "Download analysed candles (CSV)",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="chart_reader_analysis.csv",
            mime="text/csv",
        )

st.divider()
st.caption(f"Chart Reader development build · {datetime.now().strftime('%d %b %Y')}")
