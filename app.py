from datetime import datetime, date, timedelta
import json
import queue
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.data_loader import read_uploaded_csv
from engine.ai_decision import analyse_trade_setup, rank_trade_setups, train_direction_model
from engine.ai_replay import replay_ai_decisions
from engine.indicators import add_indicators
from engine.features import add_features
from engine.ml_dataset import build_ml_dataset
from engine.ml_pattern_context import analyze_pattern_context
from engine.ml_cross_timeframe import analyze_cross_timeframe_robustness
from engine.market_structure import add_market_structure
from engine.outcomes import label_forward_returns, outcome_summary
from engine.paper_trading import (
    close_paper_trade, load_trade_ledger, open_paper_trade, paper_trading_summary,
    suggest_quantity, trades_frame,
)
from engine.historical_validation import audit_historical_ohlcv, audit_nse_session_integrity, apply_nse_signal_policy
from engine.patterns import detect_patterns
from data.nifty50 import NIFTY50_SYMBOLS
from providers.fivepaisa import (
    connection_status, get_credentials, build_oauth_url, exchange_request_token,
    resolve_scrip_codes_via_market_feed, fetch_historical_candles,
)
from providers.fivepaisa_live import FivePaisaLiveFeed, LiveCandleBuilder

st.set_page_config(
    page_title="Chart Reader | Machine Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Chart Reader — Machine Intelligence")
st.caption("An evidence-led market analysis laboratory · 5paisa historical and live read-only market data")

if "fivepaisa_access_token" not in st.session_state:
    st.session_state["fivepaisa_access_token"] = ""
if "fivepaisa_client_code" not in st.session_state:
    st.session_state["fivepaisa_client_code"] = ""
credentials = get_credentials(st.secrets)
request_token = st.query_params.get("RequestToken")
if request_token and credentials["api_key"] and credentials["encryption_key"] and credentials["user_id"]:
    try:
        token, client_code = exchange_request_token(credentials, request_token)
        st.session_state["fivepaisa_access_token"] = token
        st.session_state["fivepaisa_client_code"] = client_code
        st.query_params.clear()
        st.success("5paisa authentication successful for this session.")
    except Exception as exc:
        st.error(str(exc))
        st.query_params.clear()
provider = connection_status(st.secrets, st.session_state["fivepaisa_access_token"])
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
    st.divider()
    st.subheader("5paisa read-only login")
    if credentials["api_key"] and credentials["redirect_url"]:
        oauth_url = build_oauth_url(credentials["api_key"], credentials["redirect_url"])
        st.link_button("Authenticate with 5paisa", oauth_url, use_container_width=True)
        if st.session_state["fivepaisa_access_token"]:
            st.success("Authenticated for this session")
            if st.button("Clear 5paisa session"):
                live_runtime = st.session_state.get("fivepaisa_live_runtime")
                if live_runtime and live_runtime.get("feed"):
                    live_runtime["feed"].stop()
                st.session_state["fivepaisa_live_runtime"] = None
                st.session_state["fivepaisa_access_token"] = ""
                st.session_state["fivepaisa_client_code"] = ""
                st.rerun()
        else:
            st.caption("Login takes place on 5paisa's secure page. Its RequestToken returns to the registered callback URL.")
    else:
        st.warning("Configure FIVEPAISA_API_KEY, FIVEPAISA_ENCRYPTION_KEY, FIVEPAISA_USER_ID and FIVEPAISA_REDIRECT_URL in Streamlit Secrets.")

st.subheader("Workspace")
overview, chart_tab, intelligence, ai_tab, structure_tab, historical_tab, live_tab, paper_tab, library_tab, research = st.tabs(
    ["Overview", "Chart analysis", "Intelligence", "AI Trade Desk", "Market structure", "Historical Data Centre", "Live Market", "Paper trading", "Pattern library", "Research & validation"]
)

if "paper_trades" not in st.session_state:
    st.session_state["paper_trades"] = []

data = None
error = None
if uploaded is not None:
    try:
        raw = read_uploaded_csv(uploaded)
        data = add_features(add_market_structure(detect_patterns(add_indicators(raw))))
        data = label_forward_returns(data, horizons=(3, 6, 12))
    except ValueError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"Unexpected data-processing error: {exc}"

with overview:
    a, b, c = st.columns(3)
    a.metric("5paisa connection", "Connected" if provider.connected else "Pending")
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
    st.caption("Updated against the merged GitHub main branch · Completed items are checked; live-data verification remains pending.")
    st.markdown(
        """
        **Completed**
        - [x] Public Streamlit application and secure secret-storage pattern
        - [x] CSV OHLCV ingestion and validation
        - [x] Initial indicators and explainable candle detectors
        - [x] Trend / swing structure and forward-return research labels
        - [x] **Phase 5:** Automated unit tests and expanded edge-case coverage
        - [x] **Phase 6:** Historical OHLCV data-quality audit and dashboard report

        **Next milestones**
        - [x] Manual paper-trading journal with risk-based sizing and CSV backup/restore
        - [ ] Verify 5paisa authentication and historical candles with real account data
        - [ ] Validate the paper journal with observed market prices and a documented sample
        - [ ] Live read-only monitoring and automated signal replay (only after data/timestamp verification)

        *Research only: no automatic order placement. Historical-data gap counts are estimates and should be checked against exchange sessions.*
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
        high_points = view[view["swing_high_confirmed"].notna()]
        low_points = view[view["swing_low_confirmed"].notna()]
        if not high_points.empty:
            fig.add_trace(go.Scatter(
                x=high_points["timestamp"], y=high_points["swing_high_confirmed"],
                mode="markers+text", text=high_points["swing_high_label"],
                textposition="top center", name="Confirmed swing highs"
            ))
        if not low_points.empty:
            fig.add_trace(go.Scatter(
                x=low_points["timestamp"], y=low_points["swing_low_confirmed"],
                mode="markers+text", text=low_points["swing_low_label"],
                textposition="bottom center", name="Confirmed swing lows"
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

        st.markdown("#### Engineered market features")
        feature_columns = [
            "feature_body_to_range", "feature_upper_wick_to_range",
            "feature_lower_wick_to_range", "feature_close_location",
            "feature_range_atr", "feature_relative_volume_20",
            "feature_return_3", "feature_return_6",
            "feature_sma_20_distance_pct", "feature_ema_20_distance_pct",
            "feature_distance_to_confirmed_swing_high_pct",
            "feature_distance_to_confirmed_swing_low_pct",
        ]
        available_features = [column for column in feature_columns if column in data.columns]
        latest_features = data.loc[:, ["timestamp"] + available_features].tail(10)
        st.dataframe(latest_features, use_container_width=True, hide_index=True)
        st.caption(
            "Feature values are descriptive inputs for future model research, not predictions. "
            "Relative volume compares the current candle with prior candles only; "
            "cross-timeframe feature alignment is intentionally not enabled yet."
        )



with ai_tab:
    st.markdown("### Autonomous AI Trade Decision Desk")
    st.caption("Completed-candle analysis · 5m entry + 15m context · read-only · no order placement")
    st.info(
        "The decision engine independently chooses LONG, SHORT or NO TRADE. "
        "Rule-based confluence is always visible; the ML overlay is used only when its chronological holdout gate passes."
    )

    st.markdown("#### Train the optional machine-learning overlay")
    training_file = st.file_uploader(
        "Upload a multi-symbol, multi-timeframe historical OHLCV CSV",
        type=["csv"], key="ai_training_csv",
        help="Required columns: timestamp, symbol, timeframe, open, high, low, close, volume. Include 5m and 15m history where available."
    )
    stored_history = st.session_state.get("historical_data", pd.DataFrame())
    has_stored_history = isinstance(stored_history, pd.DataFrame) and not stored_history.empty
    if has_stored_history:
        st.caption(f"Historical Data Centre data is available in this session: {len(stored_history):,} rows.")
    if st.button(
        "Train and validate ML model", key="ai_train_button", type="primary",
        disabled=training_file is None and not has_stored_history,
    ):
        try:
            training_frame = pd.read_csv(training_file) if training_file is not None else stored_history.copy()
            with st.spinner("Building causal features and testing a chronological holdout..."):
                model, report = train_direction_model(training_frame)
            st.session_state["ai_ml_model"] = model
            st.session_state["ai_ml_report"] = report
            st.session_state["ai_ml_features"] = report.get("features", [])
            if model is not None:
                st.success("ML overlay passed its holdout gate and is enabled for this session.")
            else:
                st.warning("ML overlay was not approved. The engine will continue in rule-based mode.")
        except Exception as exc:
            st.error(f"ML training could not complete: {exc}")

    ml_report = st.session_state.get("ai_ml_report")
    if ml_report:
        c1, c2, c3 = st.columns(3)
        c1.metric("ML overlay", "APPROVED" if ml_report.get("approved") else "DISABLED")
        c2.metric("Balanced accuracy", f"{ml_report['balanced_accuracy']:.1%}" if ml_report.get("balanced_accuracy") is not None else "—")
        c3.metric("Validation rows", ml_report.get("validation_rows", "—"))
        st.caption(ml_report.get("reason", ""))
        if ml_report.get("first_validation_date"):
            st.caption(
                f"Chronological holdout: {ml_report['first_validation_date']} to "
                f"{ml_report['last_validation_date']} · majority baseline "
                f"{ml_report['majority_baseline']:.1%}"
            )
    else:
        st.caption("No ML model has been trained in this session. The rule-based decision engine remains available; no ML probability will be displayed.")

    st.divider()
    st.markdown("#### Scan completed live candles")
    live_bars = st.session_state.get("fivepaisa_live_bars", {})
    five_bars = live_bars.get("5m", [])
    fifteen_bars = live_bars.get("15m", [])
    if five_bars or fifteen_bars:
        five_counts = pd.DataFrame(five_bars).groupby("symbol").size() if five_bars else pd.Series(dtype=int)
        fifteen_counts = pd.DataFrame(fifteen_bars).groupby("symbol").size() if fifteen_bars else pd.Series(dtype=int)
        readiness_symbols = sorted(set(five_counts.index) | set(fifteen_counts.index))
        readiness = pd.DataFrame([{
            "Stock": ticker,
            "Completed 5m bars": int(five_counts.get(ticker, 0)),
            "Completed 15m bars": int(fifteen_counts.get(ticker, 0)),
            "Ready": int(five_counts.get(ticker, 0)) >= 21 and int(fifteen_counts.get(ticker, 0)) >= 21,
        } for ticker in readiness_symbols])
        st.markdown("##### Per-stock candle readiness")
        st.dataframe(readiness, use_container_width=True, hide_index=True)
        ready_count = int(readiness["Ready"].sum()) if not readiness.empty else 0
        st.caption(f"{ready_count} of {len(readiness)} subscribed stocks currently have the minimum 21 completed bars on both timeframes.")
    if not five_bars or not fifteen_bars:
        st.warning("Start the 5paisa read-only feed and collect at least 21 completed candles per timeframe for each stock. The scanner will abstain until enough history is available.")
    else:
        if st.button("Analyse NIFTY 50 watchlist", key="ai_scan_live", type="primary"):
            five_frame = pd.DataFrame(five_bars)
            fifteen_frame = pd.DataFrame(fifteen_bars)
            scan_results = []
            symbols = sorted(set(five_frame.get("symbol", [])) & set(fifteen_frame.get("symbol", [])))
            model = st.session_state.get("ai_ml_model")
            report = st.session_state.get("ai_ml_report") or {}
            feature_columns = st.session_state.get("ai_ml_features", [])
            for ticker in symbols:
                five_symbol = five_frame[five_frame["symbol"] == ticker].tail(300)
                fifteen_symbol = fifteen_frame[fifteen_frame["symbol"] == ticker].tail(300)
                try:
                    scan_results.append(analyse_trade_setup(
                        five_symbol, fifteen_symbol, symbol=ticker,
                        ml_model=model, ml_feature_columns=feature_columns,
                        ml_approved=bool(report.get("approved", False)),
                    ))
                except Exception as exc:
                    scan_results.append({
                        "symbol": ticker, "decision": "NO TRADE",
                        "status": "DATA_ERROR", "reason": str(exc),
                    })
            st.session_state["ai_last_scan_results"] = scan_results
            st.session_state["ai_last_scan_time"] = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST")

        results = st.session_state.get("ai_last_scan_results", [])
        if results:
            st.caption("Last scan: " + st.session_state.get("ai_last_scan_time", ""))
            qualified = rank_trade_setups(results, limit=3)
            if qualified:
                st.success(f"{len(qualified)} qualified candidate(s) found. Review the evidence and confirm the next-bar entry.")
                st.dataframe(pd.DataFrame([{
                    "Stock": item["symbol"], "Decision": item["decision"],
                    "Reference close": item["entry_reference"], "Stop-loss": item["stop_loss"],
                    "Target": item["target"], "R:R": item["reward_risk"],
                    "Confluence score": item["score"], "ML overlay": item["ml_used"],
                } for item in qualified]), use_container_width=True, hide_index=True)
                for item in qualified:
                    with st.expander(f"{item['symbol']} · {item['decision']} · score {item['score']}"):
                        st.write("**Evidence**")
                        for reason in item["reasons"]:
                            st.write("• " + reason)
                        for warning in item["warnings"]:
                            st.caption("⚠ " + warning)
            else:
                st.info("NO TRADE: no stock passed the quality filters on this scan.")
            with st.expander("All stock decisions, including abstentions"):
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    st.caption(
        "The engine is autonomous in selecting LONG/SHORT/NO TRADE from its programmed evidence and, when approved, an ML overlay. "
        "It is not an autonomous broker or guaranteed-profit system. Live feed accuracy, candle boundaries, liquidity and next-bar confirmation remain required."
    )

    st.divider()
    st.markdown("#### Historical AI replay laboratory")
    replay_history = st.session_state.get("historical_data", pd.DataFrame())
    replay_required = {"timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"}
    if replay_history.empty or replay_required.difference(replay_history.columns):
        st.info("Fetch multi-symbol 5m and 15m candles in the Historical Data Centre first. Replay requires timestamp, symbol, timeframe and OHLCV.")
    else:
        replay_symbols = sorted(replay_history["symbol"].dropna().astype(str).unique())
        replay_symbol = st.selectbox("Replay stock", replay_symbols, key="ai_replay_symbol")
        cost_col, slip_col = st.columns(2)
        replay_cost = cost_col.number_input("Round-trip transaction cost (bps)", min_value=0.0, max_value=500.0, value=10.0, step=1.0, key="ai_replay_cost")
        replay_slippage = slip_col.number_input("Slippage per side (bps)", min_value=0.0, max_value=200.0, value=5.0, step=1.0, key="ai_replay_slippage")
        replay_cas_eligible = st.checkbox(
            "This security is eligible for NSE Closing Auction Session (CAS)",
            value=False, key="ai_replay_cas_eligible",
            help="Use the official NSE security master. Before 7 Dec 2026, continuous trading ends at 15:15 for CAS securities; from 7 Dec, it ends at 15:30.",
        )
        st.caption("Research assumptions only. Replay uses next 5m open, same-session exits, and stop-first resolution when both barriers touch in one candle. Configured special-session dates are excluded.")
        if st.button("Run historical AI replay", key="ai_replay_run", type="primary"):
            selected = replay_history[replay_history["symbol"].astype(str) == replay_symbol].copy()
            five_history = selected[selected["timeframe"].astype(str).str.lower() == "5m"].copy()
            fifteen_history = selected[selected["timeframe"].astype(str).str.lower() == "15m"].copy()
            try:
                with st.spinner(f"Replaying {replay_symbol} using completed 5m decisions and 15m context..."):
                    replay_result = replay_ai_decisions(
                        five_history, fifteen_history, symbol=replay_symbol,
                        transaction_cost_bps=replay_cost, slippage_bps=replay_slippage,
                        cas_eligible=replay_cas_eligible,
                    )
                st.session_state["ai_replay_result"] = replay_result
                st.session_state["ai_replay_symbol"] = replay_symbol
            except Exception as exc:
                st.error(f"Historical replay rejected this dataset: {exc}")
        replay_result = st.session_state.get("ai_replay_result")
        if replay_result and replay_result.get("symbol") == replay_symbol:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Closed trades", replay_result["trade_count"])
            m2.metric("Win rate", f"{replay_result['win_rate_pct']:.1f}%" if replay_result["win_rate_pct"] is not None else "—")
            m3.metric("Mean net return / trade", f"{replay_result['mean_net_return_pct']:.3f}%" if replay_result["mean_net_return_pct"] is not None else "—")
            m4.metric("Max drawdown", f"{replay_result['max_drawdown_pct']:.2f}%" if replay_result["max_drawdown_pct"] is not None else "—")
            st.caption(f"Profit factor: {replay_result['profit_factor']:.2f}" if replay_result["profit_factor"] is not None and np.isfinite(replay_result["profit_factor"]) else "Profit factor: unavailable")
            if replay_result["trades"]:
                replay_frame = pd.DataFrame(replay_result["trades"])
                st.dataframe(replay_frame, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download replay trades CSV",
                    replay_frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"{replay_symbol.lower()}_ai_replay_trades.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.caption("This is a research replay, not a validated profitability claim. Audit timestamps, session coverage, costs and out-of-sample dates before interpreting metrics.")

with structure_tab:
    st.markdown("### Market Structure Intelligence")
    if data is None:
        st.info("Upload historical OHLCV candles in the sidebar to inspect confirmed swing highs/lows, HH/HL/LH/LL labels and BOS/CHoCH events.")
    else:
        st.caption(
            "Swing points are shown only after the right-side confirmation candles have completed. "
            "Structure breaks require a candle close beyond a confirmed swing level."
        )
        a, b, c = st.columns(3)
        a.metric("Current structure bias", str(data["structure_bias"].iloc[-1]))
        b.metric("Confirmed swing highs", int(data["swing_high_confirmed"].notna().sum()))
        c.metric("Confirmed swing lows", int(data["swing_low_confirmed"].notna().sum()))
        events = data.loc[
            data["structure_event"] != "",
            ["timestamp", "close", "structure_event", "structure_break_level"]
        ].tail(30)
        st.markdown("#### Recent BOS / CHoCH events")
        if events.empty:
            st.info("No confirmed BOS/CHoCH events in this sample.")
        else:
            st.dataframe(events, use_container_width=True, hide_index=True)
        swings = data.loc[
            (data["swing_high_label"] != "") | (data["swing_low_label"] != ""),
            ["timestamp", "swing_high_label", "swing_low_label",
             "swing_high_confirmed", "swing_low_confirmed"]
        ].tail(40)
        st.markdown("#### Confirmed swing sequence")
        if swings.empty:
            st.info("Not enough candles to confirm swing points yet.")
        else:
            st.dataframe(swings, use_container_width=True, hide_index=True)
        st.caption(
            "HH/HL/LH/LL describe successive confirmed pivots. BOS/CHoCH are structural events, "
            "not standalone trade signals. Pivot confirmation intentionally introduces delay "
            "to avoid look-ahead bias."
        )

with historical_tab:
    st.markdown("### 5paisa Historical Data Centre")
    st.caption("NIFTY 50 cash equities · 5-minute and 15-minute candles · read-only market-data access")
    st.info("5paisa supports 5m and 15m historical candles. Intraday requests are limited to six months per request.")
    st.write(f"**Universe:** {len(NIFTY50_SYMBOLS)} NIFTY 50 constituents (list dated 28-Oct-2026).")
    if not st.session_state["fivepaisa_access_token"]:
        st.warning("Authenticate with 5paisa from the sidebar before requesting historical candles.")
    else:
        left, right = st.columns(2)
        start_date = left.date_input("From date", value=date.today() - timedelta(days=30), max_value=date.today())
        end_date = right.date_input("To date", value=date.today(), max_value=date.today())
        selected_symbols = st.multiselect(
            "NIFTY 50 stocks", options=NIFTY50_SYMBOLS,
            default=["RELIANCE", "HDFCBANK", "INFY"],
            help="All 50 constituents are available. Fetch up to 10 stocks per batch."
        )
        selected_intervals = st.multiselect("Candle timeframes", ["5m", "15m"], default=["5m", "15m"])
        st.caption("Maximum 10 stocks per batch (up to 20 sequential historical requests). No orders are placed.")
        if st.button("Fetch historical candles", type="primary", use_container_width=True):
            if start_date > end_date:
                st.error("From date must be on or before To date.")
            elif (end_date - start_date).days > 183:
                st.error("Choose a date range of six months or less.")
            elif not selected_symbols or not selected_intervals:
                st.error("Select at least one stock and one timeframe.")
            elif len(selected_symbols) > 10:
                st.error("Please fetch no more than 10 stocks per batch.")
            else:
                try:
                    credentials = get_credentials(st.secrets)
                    with st.spinner("Resolving NSE stock codes through 5paisa MarketFeed..."):
                        scrip_codes = resolve_scrip_codes_via_market_feed(
                            st.session_state["fivepaisa_access_token"],
                            credentials["api_key"],
                            st.session_state["fivepaisa_client_code"],
                            selected_symbols,
                        )
                    collected, failures = [], []
                    total = len(selected_symbols) * len(selected_intervals)
                    progress, status = st.progress(0), st.empty()
                    completed = 0
                    for stock in selected_symbols:
                        code = scrip_codes[stock]
                        for timeframe in selected_intervals:
                            status.write(f"Fetching {stock} · {timeframe}...")
                            try:
                                candles = fetch_historical_candles(
                                    st.session_state["fivepaisa_access_token"],
                                    code, timeframe, start_date, end_date,
                                    client_code=st.session_state["fivepaisa_client_code"],
                                )
                                candles["symbol"] = stock
                                candles["timeframe"] = timeframe
                                candles["scrip_code"] = code
                                collected.append(candles)
                            except Exception as exc:
                                failures.append(f"{stock} ({timeframe}): {exc}")
                            completed += 1
                            progress.progress(min(completed / total, 1.0))
                    status.empty()
                    st.session_state["historical_data"] = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
                    st.session_state["historical_data_failures"] = failures
                except Exception as exc:
                    st.error(str(exc))
    hist = st.session_state.get("historical_data", pd.DataFrame())
    if not hist.empty:
        st.success(f"Fetched {len(hist):,} candles across {hist['symbol'].nunique()} stocks and {hist['timeframe'].nunique()} timeframes.")
        p, q, r = st.columns(3)
        p.metric("Stocks received", int(hist["symbol"].nunique()))
        q.metric("Candle rows", f"{len(hist):,}")
        r.metric("Timeframes", ", ".join(sorted(hist["timeframe"].unique())))
        st.download_button(
            "Download raw 5paisa timestamp audit CSV",
            hist.to_csv(index=False).encode("utf-8"),
            file_name="fivepaisa_provider_timestamp_audit.csv",
            mime="text/csv",
            help="Includes provider_timestamp_raw exactly as returned, alongside parsed timestamps and OHLCV.",
            use_container_width=True,
        )
        st.caption(
            "For provider verification, fetch the same liquid NSE symbol in both 5m and 15m intervals "
            "covering at least two complete regular sessions. Download this CSV without editing it. "
            "Do not share API keys, access tokens, or login details."
        )
        chosen_symbol = st.selectbox("Preview stock", sorted(hist["symbol"].unique()))
        choices = sorted(hist.loc[hist["symbol"] == chosen_symbol, "timeframe"].unique())
        chosen_interval = st.selectbox("Preview timeframe", choices)
        preview = hist[(hist["symbol"] == chosen_symbol) & (hist["timeframe"] == chosen_interval)]
        st.dataframe(preview.tail(100), use_container_width=True, hide_index=True)

        st.markdown("#### Historical data quality audit")
        cas_symbols = st.multiselect(
            "CAS-eligible securities in this audit",
            options=sorted(hist["symbol"].unique()),
            default=[],
            help="Select only securities confirmed as CAS-eligible for the selected historical dates using the applicable NSE security list. Unselected securities use the non-CAS continuous-session schedule."
        )
        st.caption("CAS eligibility is not inferred from NIFTY 50 membership. The selection applies to all dates in this downloaded sample; do not use it across dates where the official eligibility list changed.")
        audit_rows = []
        quarantine_frames = []
        regular_signal_frames = []
        eligible_counts = []
        for (audit_symbol, audit_interval), group in hist.groupby(["symbol", "timeframe"]):
            try:
                audit = audit_historical_ohlcv(
                    group[["timestamp", "open", "high", "low", "close", "volume"]],
                    audit_interval,
                )
                session_audit = audit_nse_session_integrity(
                    group[["timestamp"]],
                    audit_interval,
                    cas_eligible=audit_symbol in cas_symbols,
                )
                # Keep the audit compatible with an older engine module that may
                # still be loaded by Streamlit during a deployment restart.
                session_classifications = session_audit.get("session_classifications", {})
                session_anomalies = set(session_audit.get("session_anomaly_indices", []))
                eligible_indices = set(session_audit.get(
                    "session_eligible_indices",
                    [
                        index for index, label in session_classifications.items()
                        if label == "regular" and index not in session_anomalies
                    ],
                ))
                classified = apply_nse_signal_policy(
                    group, audit_interval, cas_eligible=audit_symbol in cas_symbols,
                )
                classified["session_eligible"] = classified["regular_signal_eligible"]
                classified["quarantine_reason"] = classified["signal_exclusion_reason"].map({
                    "special_session_research_only": "Special session: historical research only",
                    "misaligned_or_session_anomaly": "Timestamp alignment/session anomaly",
                    "non_continuous_closing_session": "Non-continuous closing session",
                    "weekend_or_holiday": "Weekend or holiday candle",
                    "outside_recognised_exchange_session": "Outside recognised exchange session",
                    "invalid_or_unclassified_timestamp": "Invalid or unclassified timestamp",
                    "": "",
                }).fillna("Invalid or unclassified timestamp")
                quarantined = classified.loc[classified["research_only"]].copy()
                regular_signal = classified.loc[classified["regular_signal_eligible"]].copy()
                if not regular_signal.empty:
                    regular_signal_frames.append(regular_signal)
                if not quarantined.empty:
                    quarantine_frames.append(quarantined)
                eligible_counts.append({
                    "Stock": audit_symbol,
                    "Timeframe": audit_interval,
                    "Regular-session eligible rows": int(classified["regular_signal_eligible"].sum()),
                    "Quarantined rows": len(quarantined),
                })
                audit_rows.append({
                    "Stock": audit_symbol,
                    "Timeframe": audit_interval,
                    "Rows": audit["rows"],
                    "Duplicate timestamps": audit["duplicate_timestamps"],
                    "Invalid OHLC rows": audit["invalid_ohlc_rows"],
                    "Negative volume rows": audit["negative_volume_rows"],
                    "Possible missing candles": audit["estimated_missing_intraday_bars"],
                    "Weekend candles": session_audit["weekend_timestamps"],
                    "Special-session candles": session_audit["special_session_timestamps"],
                    "CAS eligible": session_audit["cas_eligible"],
                    "CAS candles": session_audit["closing_auction_session_timestamps"],
                    "CAS transition candles": session_audit["closing_transition_timestamps"],
                    "Closing-session candles": session_audit["closing_session_timestamps"],
                    "Outside exchange sessions": session_audit["outside_session_timestamps"],
                    "Misaligned timestamps": session_audit["misaligned_timestamps"],
                    "Non-regular-session rows": session_audit["non_regular_session_rows"],
                    "Session anomaly rows": session_audit["session_anomaly_rows"],
                    "Quality status": (
                        "Session review required" if session_audit["session_anomaly_rows"]
                        else ("Non-regular session data" if session_audit["non_regular_session_rows"] else audit["quality_status"])
                    ),
                    "Warnings": " | ".join(audit["warnings"] + session_audit["warnings"]),
                })
            except ValueError as exc:
                audit_rows.append({
                    "Stock": audit_symbol,
                    "Timeframe": audit_interval,
                    "Rows": len(group),
                    "Quality status": "Audit error",
                    "Warnings": str(exc),
                })
        audit_frame = pd.DataFrame(audit_rows)
        st.dataframe(audit_frame, use_container_width=True, hide_index=True)
        st.markdown("#### Session eligibility gate")
        st.caption(
            "Only aligned regular-session rows are session-eligible. Muhurat candles are retained for research, while misaligned, invalid, weekend/holiday, closing-auction, transition and closing-session rows are quarantined. This is a session gate only; OHLCV integrity checks remain separate. Raw candles are never deleted or rewritten."
        )
        st.dataframe(pd.DataFrame(eligible_counts), use_container_width=True, hide_index=True)
        quarantine_frame = (
            pd.concat(quarantine_frames, ignore_index=False).sort_values(["symbol", "timeframe", "timestamp"])
            if quarantine_frames else pd.DataFrame(columns=list(hist.columns) + ["session_classification", "session_eligible", "quarantine_reason"])
        )
        st.metric("Quarantined rows (preserved)", len(quarantine_frame))
        if not quarantine_frame.empty:
            st.dataframe(quarantine_frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download quarantined candles (CSV)",
            data=quarantine_frame.to_csv(index=False).encode("utf-8"),
            file_name="chart_reader_quarantined_candles.csv",
            mime="text/csv", use_container_width=True,
        )
        regular_signal_frame = (
            pd.concat(regular_signal_frames, ignore_index=True)
            if regular_signal_frames else pd.DataFrame(columns=list(hist.columns) + [
                "session_classification", "regular_signal_eligible", "research_only",
                "signal_exclusion_reason", "session_eligible", "quarantine_reason",
            ])
        )
        st.metric("Regular-session signal-eligible rows", len(regular_signal_frame))
        st.download_button(
            "Download regular-session signal candles (CSV)",
            data=regular_signal_frame.to_csv(index=False).encode("utf-8"),
            file_name="chart_reader_regular_signal_candles.csv",
            mime="text/csv", use_container_width=True,
        )
        st.caption(
            "Missing-candle counts are estimates within the same date. "
            "Recognised sessions include the configured 8-Nov-2026 Muhurat continuous session. Session timings are date-aware: before 7-Dec-2026, CAS-eligible securities use continuous trading until 15:15, CAS 15:15–15:35, transition 15:35–15:50, and post-close 15:50–16:00; from 7-Dec-2026, continuous trading runs until 15:30, CAS 15:30–15:45, transition 15:45–15:50, and post-close 15:50–16:00. Non-CAS securities continue until 15:30, then are classified in the transition period. CAS eligibility must be confirmed against the dated NSE security list. The audit assumes candle-start timestamps; verify 5paisa timestamp semantics before strategy use. Holiday calendars must be updated from official NSE circulars."
        )
        st.download_button(
            "Download data quality audit (CSV)",
            data=audit_frame.to_csv(index=False).encode("utf-8"),
            file_name="chart_reader_data_quality_audit.csv",
            mime="text/csv", use_container_width=True
        )
        st.download_button(
            "Download fetched NIFTY 50 candles (CSV)",
            data=hist.to_csv(index=False).encode("utf-8"),
            file_name="chart_reader_nifty50_historical_ohlcv.csv",
            mime="text/csv", use_container_width=True
        )
    failures = st.session_state.get("historical_data_failures", [])
    if failures:
        with st.expander(f"Fetch warnings ({len(failures)})"):
            for item in failures:
                st.write("• " + item)
    st.caption("NIFTY 50 membership is dated and should be refreshed after NSE reviews. Actual historical availability depends on 5paisa account permissions and API response.")
    
with live_tab:
    st.markdown("### 5paisa Live Market Monitor")
    st.caption("Authenticated NSE cash ticks · completed 5-minute and 15-minute candles · read-only")
    st.info("This monitor displays live market data only. It does not generate validated ML recommendations or place orders.")
    if not st.session_state["fivepaisa_access_token"]:
        st.warning("Authenticate with 5paisa from the sidebar before starting the live feed.")
    else:
        selected_live_symbols = st.multiselect(
            "NIFTY 50 live watchlist", options=NIFTY50_SYMBOLS,
            default=["RELIANCE", "HDFCBANK", "INFY"],
            max_selections=50,
            help="Choose up to 50 NSE cash equities. Start the feed manually; no subscriptions are made automatically."
        )
        runtime = st.session_state.get("fivepaisa_live_runtime")
        running = bool(runtime and runtime.get("feed"))
        start_col, stop_col = st.columns(2)
        start_clicked = start_col.button(
            "Start live feed", type="primary", use_container_width=True,
            disabled=running or not selected_live_symbols,
        )
        stop_clicked = stop_col.button(
            "Stop live feed", use_container_width=True, disabled=not running,
        )
        if start_clicked:
            try:
                credentials = get_credentials(st.secrets)
                with st.spinner("Resolving selected NSE symbols through 5paisa..."):
                    scrip_codes = resolve_scrip_codes_via_market_feed(
                        st.session_state["fivepaisa_access_token"],
                        credentials["api_key"],
                        st.session_state["fivepaisa_client_code"],
                        selected_live_symbols,
                    )
                event_queue = queue.Queue(maxsize=10000)
                builders = {"5m": LiveCandleBuilder(5), "15m": LiveCandleBuilder(15)}

                def publish_live_event(event):
                    try:
                        event_queue.put_nowait(event)
                    except queue.Full:
                        try:
                            event_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            event_queue.put_nowait(event)
                        except queue.Full:
                            pass

                def handle_live_tick(tick):
                    publish_live_event({"kind": "tick", "data": tick})
                    for timeframe, builder in builders.items():
                        completed_bar = builder.add_tick(tick)
                        if completed_bar is not None:
                            publish_live_event({
                                "kind": "candle", "timeframe": timeframe,
                                "data": completed_bar,
                            })

                feed = FivePaisaLiveFeed(
                    st.session_state["fivepaisa_access_token"],
                    st.session_state["fivepaisa_client_code"],
                    credentials["api_key"], scrip_codes,
                    on_tick=handle_live_tick,
                    on_error=lambda message: publish_live_event({"kind": "error", "message": message}),
                    on_status=lambda message: publish_live_event({"kind": "status", "message": message}),
                )
                feed.start()
                st.session_state["fivepaisa_live_runtime"] = {
                    "feed": feed, "queue": event_queue,
                    "symbols": list(scrip_codes), "status": "Connecting",
                }
                st.session_state["fivepaisa_live_ticks"] = []
                st.session_state["fivepaisa_live_bars"] = {"5m": [], "15m": []}
                st.session_state["fivepaisa_live_error"] = ""
                st.success(f"Live feed starting for {len(scrip_codes)} selected NSE stocks.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not start 5paisa live feed: {exc}")
        if stop_clicked and runtime:
            runtime["feed"].stop()
            st.session_state["fivepaisa_live_runtime"] = None
            st.info("Live feed stopped.")
            st.rerun()

        runtime = st.session_state.get("fivepaisa_live_runtime")
        if runtime:
            @st.fragment(run_every="1s")
            def render_live_feed_panel():
                live_runtime = st.session_state.get("fivepaisa_live_runtime")
                if not live_runtime:
                    st.info("Live feed is stopped.")
                    return
                event_queue = live_runtime["queue"]
                while True:
                    try:
                        event = event_queue.get_nowait()
                    except queue.Empty:
                        break
                    if event["kind"] == "tick":
                        tick = event["data"]
                        live_runtime["status"] = (
                            "Receiving live ticks · latest "
                            + pd.Timestamp(tick["timestamp"]).strftime("%H:%M:%S IST")
                        )
                        st.session_state["fivepaisa_live_ticks"].append(tick)
                        st.session_state["fivepaisa_live_ticks"] = st.session_state["fivepaisa_live_ticks"][-300:]
                    elif event["kind"] == "candle":
                        timeframe = event["timeframe"]
                        bars = st.session_state["fivepaisa_live_bars"][timeframe]
                        bar = event["data"]
                        bars.append(bar)
                        # Keep a rolling history per symbol, not one shared
                        # 300-event buffer for the entire watchlist.
                        symbol_indices = [
                            index for index, item in enumerate(bars)
                            if item.get("symbol") == bar.get("symbol")
                        ]
                        if len(symbol_indices) > 300:
                            del bars[symbol_indices[0]]
                    elif event["kind"] == "status":
                        live_runtime["status"] = event["message"]
                    elif event["kind"] == "error":
                        st.session_state["fivepaisa_live_error"] = event["message"]
                st.metric("Feed status", live_runtime["status"])
                st.caption("Subscribed: " + ", ".join(live_runtime["symbols"]))
                if st.session_state.get("fivepaisa_live_error"):
                    st.warning(st.session_state["fivepaisa_live_error"])
                ticks = st.session_state.get("fivepaisa_live_ticks", [])
                if ticks:
                    latest = pd.DataFrame(ticks).sort_values("timestamp", ascending=False)
                    st.markdown("#### Latest live ticks")
                    st.dataframe(
                        latest[["timestamp", "symbol", "last_price", "last_qty", "total_qty"]].head(30),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("Connected or connecting. Waiting for the first market tick.")
                timeframe = st.selectbox("Completed candle timeframe", ["5m", "15m"], key="live_candle_timeframe")
                bars = st.session_state.get("fivepaisa_live_bars", {}).get(timeframe, [])
                if bars:
                    st.markdown(f"#### Completed {timeframe} candles")
                    st.dataframe(
                        pd.DataFrame(bars).sort_values(["symbol", "timestamp"], ascending=[True, False]),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption("Completed candles will appear after the next interval begins. In-progress candles are not displayed as completed bars.")
            render_live_feed_panel()
        else:
            st.caption("The live feed is stopped. Select a watchlist and start it when ready.")


with paper_tab:
    st.markdown("### Paper Trading — Manual Simulation")
    st.warning(
        "Simulation only. This module does not place orders or fetch live quotes. "
        "Enter prices observed from your live market screen; historical CSV closes are not live prices."
    )
    st.caption(
        "Start with the five-stock validation group: RELIANCE, HDFCBANK, ICICIBANK, "
        "ADANIENT and ADANIPORTS. No automated strategy signals are generated."
    )

    cfg_left, cfg_mid, cfg_right = st.columns(3)
    virtual_capital = cfg_left.number_input(
        "Virtual starting capital (₹)", min_value=1000.0, value=100000.0, step=5000.0,
        key="paper_virtual_capital"
    )
    risk_pct = cfg_mid.number_input(
        "Risk budget per trade (%)", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        key="paper_risk_pct",
        help="Position size is capped using the distance from entry to stop."
    )
    friction_bps = cfg_right.number_input(
        "Estimated round-trip friction (bps)", min_value=0.0, max_value=100.0,
        value=10.0, step=1.0, key="paper_friction_bps",
        help="A simple estimate for costs and slippage, not a broker fee/tax calculation."
    )
    st.caption("Controls: maximum 3 simultaneous positions, maximum 3% combined open stop-risk, one open trade per stock, and cash-only notional sizing.")

    trades = st.session_state["paper_trades"]
    summary = paper_trading_summary(trades, virtual_capital)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Closed trades", summary["closed_trades"])
    m2.metric("Open trades", summary["open_trades"])
    m3.metric("Realised net P&L", f"₹{summary['net_pnl']:,.2f}")
    m4.metric("Win rate", "—" if summary["win_rate_pct"] is None else f"{summary['win_rate_pct']:.1f}%")

    st.markdown("#### Record a paper setup")
    watchlist = ["RELIANCE", "HDFCBANK", "ICICIBANK", "ADANIENT", "ADANIPORTS"]
    c1, c2, c3 = st.columns(3)
    paper_symbol = c1.selectbox("Stock", watchlist, key="paper_symbol")
    paper_timeframe = c2.selectbox("Timeframe", ["5m", "15m"], key="paper_timeframe")
    paper_side = c3.selectbox("Direction", ["LONG", "SHORT"], key="paper_side")
    p1, p2, p3 = st.columns(3)
    entry_price = p1.number_input("Observed entry price (₹)", min_value=0.0, value=0.0, step=0.05, key="paper_entry", help="Enter the current observed market price.")
    stop_price = p2.number_input("Stop-loss price (₹)", min_value=0.0, value=0.0, step=0.05, key="paper_stop")
    target_price = p3.number_input("Target price (₹)", min_value=0.0, value=0.0, step=0.05, key="paper_target")
    suggested_qty = 0
    sizing_ok = False
    if min(entry_price, stop_price, target_price) <= 0:
        st.info("Enter observed entry, stop-loss and target prices to calculate position size.")
    else:
        try:
            suggested_qty = suggest_quantity(
                virtual_capital, risk_pct, entry_price, stop_price
            )
            risk_rupees = abs(entry_price - stop_price) * suggested_qty
            reward_rupees = abs(target_price - entry_price) * suggested_qty
            st.info(
                f"Suggested quantity: {suggested_qty} shares · "
                f"Planned stop-risk: ₹{risk_rupees:,.2f} · "
                f"Target reward before costs: ₹{reward_rupees:,.2f} · "
                f"Reward/risk: {reward_rupees / risk_rupees:.2f}R"
            )
            sizing_ok = (
                (paper_side == "LONG" and stop_price < entry_price < target_price)
                or (paper_side == "SHORT" and target_price < entry_price < stop_price)
            )
            if not sizing_ok:
                st.error("Price levels do not match the selected direction. Correct entry, stop and target.")
        except ValueError as exc:
            st.error(f"Position sizing unavailable: {exc}")
    setup_notes = st.text_input(
        "Setup reason / confirmation (required)",
        placeholder="Example: 15m uptrend, 5m pullback, volume confirmation",
        key="paper_setup_notes"
    )
    confirm_open = st.checkbox("I confirm these are observed prices and this is a simulated trade.", key=f"paper_confirm_open_{len(st.session_state['paper_trades'])}")
    if st.button("Open paper trade", type="primary", disabled=(suggested_qty < 1 or not sizing_ok or not setup_notes.strip() or not confirm_open), use_container_width=True):
        try:
            new_trade = open_paper_trade(
                st.session_state["paper_trades"],
                symbol=paper_symbol,
                timeframe=paper_timeframe,
                side=paper_side,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                quantity=suggested_qty,
                capital=virtual_capital,
                max_open_trades=3,
                max_total_open_risk_pct=3.0,
                max_total_exposure_pct=100.0,
                estimated_cost_bps=friction_bps,
                notes=setup_notes,
            )
            st.session_state["paper_trades"].append(new_trade)
            st.success(f"Paper trade {new_trade['trade_id']} opened. No broker order was sent.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    open_positions = [t for t in st.session_state["paper_trades"] if t.get("status") == "OPEN"]
    if open_positions:
        st.markdown("#### Close a paper position")
        trade_labels = {
            f"{t['trade_id']} · {t['symbol']} · {t['side']} · {t['quantity']} shares": t["trade_id"]
            for t in open_positions
        }
        selected_trade_label = st.selectbox("Open position", list(trade_labels), key="paper_close_trade")
        selected_trade_id = trade_labels[selected_trade_label]
        exit_price = st.number_input("Observed exit price (₹)", min_value=0.01, value=0.01, step=0.05, key="paper_exit")
        confirm_close = st.checkbox("I confirm this is the observed simulated exit price.", key=f"paper_confirm_close_{selected_trade_id}")
        if st.button("Close selected paper trade", disabled=not confirm_close, use_container_width=True):
            try:
                for index, trade in enumerate(st.session_state["paper_trades"]):
                    if trade["trade_id"] == selected_trade_id:
                        st.session_state["paper_trades"][index] = close_paper_trade(trade, exit_price)
                        break
                st.success("Paper position closed and simulated P&L recorded.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.markdown("#### Restore a previous paper ledger")
    ledger_upload = st.file_uploader(
        "Upload a previously downloaded paper-trading ledger CSV",
        type=["csv"],
        key="paper_ledger_restore",
    )
    restore_confirm = st.checkbox("I understand restoring will replace the current in-session ledger.", key="paper_restore_confirm")
    if ledger_upload is not None and st.button("Restore uploaded ledger", key="restore_paper_ledger", disabled=not restore_confirm):
        try:
            restored = load_trade_ledger(ledger_upload)
            st.session_state["paper_trades"] = restored
            st.success(f"Restored {len(restored)} paper-trade records.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.markdown("#### Paper trade ledger")
    ledger = trades_frame(st.session_state["paper_trades"])
    if ledger.empty:
        st.info("No paper trades recorded yet. Record a setup above to begin.")
    else:
        st.dataframe(ledger, use_container_width=True, hide_index=True)
        st.download_button(
            "Download paper-trading ledger (CSV)",
            data=ledger.to_csv(index=False).encode("utf-8"),
            file_name="chart_reader_paper_trading_ledger.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "The ledger is held in this Streamlit session. Download it regularly as a backup; "
            "a server restart or new session may clear session data. Use the restore control above to reload a saved CSV."
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("Profit factor", "—" if summary["profit_factor"] is None else ("∞" if summary["profit_factor"] == float("inf") else f"{summary['profit_factor']:.2f}"))
        s2.metric("Average R", "—" if summary["average_r"] is None else f"{summary['average_r']:.2f}R")
        s3.metric("Max realised drawdown", f"{summary['max_drawdown_pct']:.2f}%")

    st.info(
        "Paper-trading protocol: log the setup before the outcome is known; use the observed market price; "
        "never move the stop farther away; close or update the journal only after the simulated exit occurs. "
        "This first release is a manual journal, not an automated live execution simulator."
    )


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
    st.markdown("### Phase 8.9 · Pattern contribution & context study")
    st.caption(
        "Separate validation-only research run using the Phase 8.9 ML dataset and analysis modules. "
        "This upload is processed in the current Streamlit session and is not committed to GitHub."
    )
    st.warning(
        "Use the raw OHLCV export with timestamp, symbol, timeframe, open, high, low, close and volume. "
        "This study reports descriptive associations only; it does not generate buy/sell signals."
    )
    pattern_context_upload = st.file_uploader(
        "Upload historical OHLCV CSV for Phase 8.9",
        type=["csv"],
        key="phase_8_9_context_csv",
    )
    pc1, pc2, pc3 = st.columns(3)
    neutral_band = pc1.selectbox(
        "Neutral band (%)", [0.10, 0.15, 0.20, 0.30], index=2,
        key="phase_8_9_neutral_band",
    )
    horizons = pc2.multiselect(
        "Forward horizons (candles)", [3, 6, 12], default=[3, 6, 12],
        key="phase_8_9_horizons",
    )
    min_support = pc3.number_input(
        "Minimum rows per comparison group", min_value=1, max_value=10000,
        value=30, step=5, key="phase_8_9_min_support",
    )
    if st.button(
        "Run Phase 8.9 validation study",
        type="primary",
        use_container_width=True,
        disabled=(pattern_context_upload is None or not horizons),
        key="run_phase_8_9_context",
    ):
        try:
            with st.spinner("Building leakage-aware features and validation-only pattern comparisons..."):
                pattern_context_upload.seek(0)
                phase89_source = pd.read_csv(pattern_context_upload)
                phase89_dataset, phase89_summary = build_ml_dataset(
                    phase89_source,
                    neutral_bands_pct={int(h): float(neutral_band) for h in horizons},
                )
                phase89_report = analyze_pattern_context(
                    phase89_dataset,
                    horizons=tuple(int(h) for h in horizons),
                    min_support=int(min_support),
                )
                phase89_report["dataset_summary"] = phase89_summary
                st.session_state["phase_8_9_context_report"] = phase89_report
                st.session_state["phase_8_9_context_error"] = ""
        except Exception as exc:
            st.session_state["phase_8_9_context_error"] = str(exc)
            st.session_state.pop("phase_8_9_context_report", None)

    phase89_error = st.session_state.get("phase_8_9_context_error", "")
    if phase89_error:
        st.error(f"Phase 8.9 could not run: {phase89_error}")
    phase89_report = st.session_state.get("phase_8_9_context_report")
    if phase89_report:
        st.success(
            f"Completed validation-only study · {phase89_report['validation_rows']:,} validation rows · "
            f"{len(phase89_report['pattern_columns'])} detected pattern features"
        )
        phase89_summary = phase89_report["dataset_summary"]
        st.json(phase89_summary["splits"])
        report_horizons = list(phase89_report["horizons"].keys())
        selected_report_horizon = st.selectbox(
            "Inspect horizon (candles)", report_horizons,
            key="phase_8_9_report_horizon",
        )
        horizon_result = phase89_report["horizons"][selected_report_horizon]
        pattern_rows = []
        for pattern_name, item in horizon_result["patterns"].items():
            overall = item["overall"]
            rates = overall.get("pattern_class_rates", {})
            differences = overall.get("rate_difference_pattern_minus_comparison", {})
            pattern_rows.append({
                "Pattern feature": pattern_name,
                "Occurrences": item["occurrences"],
                "Occurrence rate (%)": round(item["occurrence_rate"] * 100, 2),
                "Eligible comparison": overall["eligible"],
                "Up rate when present (%)": round(rates.get("1", float("nan")) * 100, 2) if rates else None,
                "Neutral rate when present (%)": round(rates.get("0", float("nan")) * 100, 2) if rates else None,
                "Down rate when present (%)": round(rates.get("-1", float("nan")) * 100, 2) if rates else None,
                "Up-rate difference (pp)": round(differences.get("1", float("nan")) * 100, 2) if differences else None,
            })
        st.dataframe(
            pd.DataFrame(pattern_rows).sort_values("Occurrences", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Rate differences are pattern-present minus pattern-absent, in percentage points. "
            "Ineligible comparisons lack the minimum support in at least one group."
        )
        with st.expander("Per-symbol and trend/volume context details"):
            selected_pattern = st.selectbox(
                "Pattern to inspect",
                list(horizon_result["patterns"].keys()),
                key="phase_8_9_selected_pattern",
            )
            selected_pattern_result = horizon_result["patterns"][selected_pattern]
            symbol_rows = []
            for symbol_name, comparison in selected_pattern_result["by_symbol"].items():
                rates = comparison.get("pattern_class_rates", {})
                symbol_rows.append({
                    "Symbol": symbol_name,
                    "Pattern rows": comparison["pattern_rows"],
                    "Comparison rows": comparison["comparison_rows"],
                    "Eligible": comparison["eligible"],
                    "Up rate when present (%)": round(rates.get("1", float("nan")) * 100, 2) if rates else None,
                })
            st.markdown("**Per-symbol comparison**")
            st.dataframe(pd.DataFrame(symbol_rows), use_container_width=True, hide_index=True)
            context_rows = []
            for context_name, comparison in selected_pattern_result["by_context"].items():
                rates = comparison.get("pattern_class_rates", {})
                context_rows.append({
                    "Trend / volume context": context_name,
                    "Pattern rows": comparison["pattern_rows"],
                    "Comparison rows": comparison["comparison_rows"],
                    "Eligible": comparison["eligible"],
                    "Up rate when present (%)": round(rates.get("1", float("nan")) * 100, 2) if rates else None,
                })
            st.markdown("**Trend and volume interaction groups**")
            if context_rows:
                st.dataframe(pd.DataFrame(context_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No rows have both trend and volume context available.")
        st.download_button(
            "Download Phase 8.9 JSON report",
            data=json.dumps(phase89_report, indent=2, allow_nan=False).encode("utf-8"),
            file_name="phase_8_9_pattern_context.json",
            mime="application/json",
            use_container_width=True,
        )
        st.info(
            "Research guardrails: only validation rows are used for outcome comparisons; "
            "train/test outcomes are excluded; subgroups below minimum support are not ranked; "
            "correlated pattern flags are not independent observations. No profitability or "
            "trading-edge conclusion is produced."
        )


    st.divider()
    st.markdown("### Phase 8.10 · Cross-timeframe robustness")
    st.caption(
        "Compare two Phase 8.9 JSON reports. This compares validation-level aggregates only; "
        "it does not synchronize individual 5m and 15m candles."
    )
    st.warning(
        "Upload one 5m report and one 15m report using the same neutral-band settings. "
        "Unsupported comparisons remain unranked. Results are descriptive, not trading signals."
    )
    cross_5m_upload = st.file_uploader(
        "Phase 8.9 report · 5-minute timeframe",
        type=["json"],
        key="phase_8_10_5m_json",
    )
    cross_15m_upload = st.file_uploader(
        "Phase 8.9 report · 15-minute timeframe",
        type=["json"],
        key="phase_8_10_15m_json",
    )
    min_effect_pp = st.slider(
        "Minimum directional effect to classify (percentage points)",
        min_value=0.0, max_value=10.0, value=2.0, step=0.5,
        key="phase_8_10_min_effect_pp",
    )
    if st.button(
        "Run Phase 8.10 cross-timeframe study",
        type="primary",
        use_container_width=True,
        disabled=(cross_5m_upload is None or cross_15m_upload is None),
        key="run_phase_8_10_cross_timeframe",
    ):
        try:
            report_5m = json.loads(cross_5m_upload.getvalue().decode("utf-8"))
            report_15m = json.loads(cross_15m_upload.getvalue().decode("utf-8"))
            cross_report = analyze_cross_timeframe_robustness(
                report_5m, report_15m, min_effect_pp=float(min_effect_pp)
            )
            st.session_state["phase_8_10_cross_report"] = cross_report
            st.session_state["phase_8_10_cross_error"] = ""
        except Exception as exc:
            st.session_state.pop("phase_8_10_cross_report", None)
            st.session_state["phase_8_10_cross_error"] = str(exc)

    cross_error = st.session_state.get("phase_8_10_cross_error", "")
    if cross_error:
        st.error(f"Phase 8.10 could not run: {cross_error}")
    cross_report = st.session_state.get("phase_8_10_cross_report")
    if cross_report:
        st.success(
            f"Completed aggregate comparison · {cross_report['common_symbol_count']} common symbols · "
            f"{cross_report['common_pattern_count']} common patterns"
        )
        st.caption(
            "Directional effect = change in (up-class rate minus down-class rate) for pattern-present "
            "versus pattern-absent rows. Same direction is not evidence of predictive skill."
        )
        selected_cross_horizon = st.selectbox(
            "Cross-timeframe horizon (candles)",
            list(cross_report["horizons"].keys()),
            key="phase_8_10_horizon",
        )
        cross_patterns = cross_report["horizons"][selected_cross_horizon]["patterns"]
        cross_rows = []
        for pattern_name, item in cross_patterns.items():
            overall = item["overall"]
            cross_rows.append({
                "Pattern": pattern_name,
                "5m effect (pp)": overall["5m"]["effect_pp"],
                "5m direction": overall["5m"]["direction"],
                "15m effect (pp)": overall["15m"]["effect_pp"],
                "15m direction": overall["15m"]["direction"],
                "Cross-timeframe status": overall["status"],
                "Eligible common symbols": item["per_symbol"]["eligible_common_symbols"],
                "Same-direction symbols": item["per_symbol"]["same_direction_symbols"],
            })
        cross_table = pd.DataFrame(cross_rows)
        if not cross_table.empty:
            st.dataframe(
                cross_table.sort_values(
                    ["Cross-timeframe status", "Same-direction symbols"],
                    ascending=[True, False],
                ),
                use_container_width=True, hide_index=True,
            )
        with st.expander("Inspect per-pattern context and symbol details"):
            selected_cross_pattern = st.selectbox(
                "Pattern",
                list(cross_patterns),
                key="phase_8_10_selected_pattern",
            )
            selected_cross = cross_patterns[selected_cross_pattern]
            st.markdown("**Per-symbol cross-timeframe comparisons**")
            symbol_rows = []
            for symbol_name, pair in selected_cross["per_symbol"]["comparisons"].items():
                symbol_rows.append({
                    "Symbol": symbol_name,
                    "5m direction": pair["5m"]["direction"],
                    "15m direction": pair["15m"]["direction"],
                    "Status": pair["status"],
                })
            st.dataframe(pd.DataFrame(symbol_rows), use_container_width=True, hide_index=True)
            st.markdown("**Trend / volume context comparisons**")
            context_rows = [
                {
                    "Context": context_name,
                    "5m direction": pair["5m"]["direction"],
                    "15m direction": pair["15m"]["direction"],
                    "Status": pair["status"],
                }
                for context_name, pair in selected_cross["by_context"].items()
            ]
            if context_rows:
                st.dataframe(pd.DataFrame(context_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No shared trend/volume context groups are available.")
        st.download_button(
            "Download Phase 8.10 JSON report",
            data=json.dumps(cross_report, indent=2, allow_nan=False).encode("utf-8"),
            file_name="phase_8_10_cross_timeframe.json",
            mime="application/json",
            use_container_width=True,
        )
        st.info(
            "Research-only: reports compare validation aggregates, not matched candle events. "
            "Short history, sparse pattern counts and correlated observations limit conclusions. "
            "No pattern is promoted to a live signal."
        )


    st.divider()
    st.markdown("### Phase 8.11 · Event-level 5m / closed-15m alignment")
    st.caption("Build a reproducible event dataset from historical and regular-signal CSV exports. The 15m context is joined only after its candle has closed. No model is trained.")
    st.warning("Timestamp handling assumes candle-start timestamps, inferred from the supplied data. Confirm this against 5paisa provider documentation before live use. Upload all batches for each export type; historical-only rows are quarantined.")
    p811_hist_files = st.file_uploader("Historical OHLCV CSV batches", type=["csv"], accept_multiple_files=True, key="phase_8_11_historical_csvs")
    p811_regular_files = st.file_uploader("Regular-signal candle CSV batches", type=["csv"], accept_multiple_files=True, key="phase_8_11_regular_csvs")
    p811_exclude_date = st.text_input("Partial/incomplete session date to exclude (YYYY-MM-DD)", value="2026-11-17", key="phase_8_11_exclude_date")
    p811_max_age = st.number_input("Maximum fresh 15m context age (minutes)", min_value=0, max_value=120, value=25, step=5, key="phase_8_11_max_context_age")
    if st.button("Build Phase 8.11 event dataset", type="primary", use_container_width=True, disabled=(not p811_hist_files or not p811_regular_files), key="run_phase_8_11_alignment"):
        try:
            from engine.ml_event_alignment import prepare_phase_8_11
            p811_hist = pd.concat([pd.read_csv(file) for file in p811_hist_files], ignore_index=True)
            p811_regular = pd.concat([pd.read_csv(file) for file in p811_regular_files], ignore_index=True)
            excluded = [p811_exclude_date.strip()] if p811_exclude_date.strip() else []
            p811_events, p811_quarantine, p811_manifest = prepare_phase_8_11(p811_hist, p811_regular, exclude_dates=excluded, max_context_age_minutes=int(p811_max_age))
            st.session_state["phase_8_11_events"] = p811_events
            st.session_state["phase_8_11_quarantine"] = p811_quarantine
            st.session_state["phase_8_11_manifest"] = p811_manifest
            st.session_state["phase_8_11_error"] = ""
        except Exception as exc:
            st.session_state.pop("phase_8_11_events", None)
            st.session_state["phase_8_11_error"] = str(exc)
    if st.session_state.get("phase_8_11_error"):
        st.error(f"Phase 8.11 could not run: {st.session_state['phase_8_11_error']}")
    p811_manifest = st.session_state.get("phase_8_11_manifest")
    if p811_manifest:
        st.success(f"Prepared {p811_manifest['eligible_5m_event_rows']:,} 5m events · {p811_manifest['events_with_closed_15m_context']:,} with closed 15m context")
        st.json(p811_manifest)
        st.download_button("Download Phase 8.11 event-level CSV", data=st.session_state["phase_8_11_events"].to_csv(index=False).encode("utf-8"), file_name="phase_8_11_event_level_5m.csv", mime="text/csv", use_container_width=True)
        st.download_button("Download historical-only quarantine CSV", data=st.session_state["phase_8_11_quarantine"].to_csv(index=False).encode("utf-8"), file_name="phase_8_11_historical_only_quarantine.csv", mime="text/csv", use_container_width=True)
        st.download_button("Download Phase 8.11 audit manifest", data=json.dumps(p811_manifest, indent=2, allow_nan=False).encode("utf-8"), file_name="phase_8_11_audit_manifest.json", mime="application/json", use_container_width=True)
        st.info("Research-only data preparation. Missing candles are not synthesized; stale higher-timeframe context is flagged; no model, trade signal or order is created.")

st.divider()
st.caption(f"Chart Reader development build · {datetime.now().strftime('%d %b %Y')}")
