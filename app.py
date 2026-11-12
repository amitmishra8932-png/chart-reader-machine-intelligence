from datetime import datetime, date, timedelta
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.data_loader import read_uploaded_csv
from engine.indicators import add_indicators
from engine.market_structure import add_market_structure
from engine.outcomes import label_forward_returns, outcome_summary
from engine.paper_trading import (
    close_paper_trade, load_trade_ledger, open_paper_trade, paper_trading_summary,
    suggest_quantity, trades_frame,
)
from engine.historical_validation import audit_historical_ohlcv, audit_nse_session_integrity
from engine.patterns import detect_patterns
from data.nifty50 import NIFTY50_SYMBOLS
from providers.fivepaisa import (
    connection_status, get_credentials, build_oauth_url, exchange_request_token,
    resolve_scrip_codes_via_market_feed, fetch_historical_candles,
)

st.set_page_config(
    page_title="Chart Reader | Machine Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Chart Reader — Machine Intelligence")
st.caption("An evidence-led market analysis laboratory · 5paisa integration in progress")

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
                st.session_state["fivepaisa_access_token"] = ""
                st.session_state["fivepaisa_client_code"] = ""
                st.rerun()
        else:
            st.caption("Login takes place on 5paisa's secure page. Its RequestToken returns to the registered callback URL.")
    else:
        st.warning("Configure FIVEPAISA_API_KEY, FIVEPAISA_ENCRYPTION_KEY, FIVEPAISA_USER_ID and FIVEPAISA_REDIRECT_URL in Streamlit Secrets.")

st.subheader("Workspace")
overview, chart_tab, intelligence, structure_tab, historical_tab, paper_tab, library_tab, research = st.tabs(
    ["Overview", "Chart analysis", "Intelligence", "Market structure", "Historical Data Centre", "Paper trading", "Pattern library", "Research & validation"]
)

if "paper_trades" not in st.session_state:
    st.session_state["paper_trades"] = []

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
    entry_price = p1.number_input("Observed entry price (₹)", min_value=0.01, value=0.01, step=0.05, key="paper_entry")
    stop_price = p2.number_input("Stop-loss price (₹)", min_value=0.01, value=0.01, step=0.05, key="paper_stop")
    target_price = p3.number_input("Target price (₹)", min_value=0.01, value=0.01, step=0.05, key="paper_target")
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
        suggested_qty = 0
        st.error(f"Position sizing unavailable: {exc}")
        sizing_ok = False
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
st.caption(f"Chart Reader development build · {datetime.now().strftime('%d %b %Y')}")
