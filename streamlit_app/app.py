"""Self-driving Streamlit demo: runs the real trading pipeline from inside
the page itself, for free hosting on Streamlit Community Cloud.

Streamlit Cloud sleeps an app once it has zero connected browser sessions,
and doesn't run background processes independent of a page view -- so
unlike alpaca_quant_agent/serve.py (a real always-on daemon for a VPS), this
can only "trade continuously" for as long as at least one browser tab stays
open on the deployed URL. `streamlit_autorefresh` forces a periodic rerun of
this whole script, and each rerun is one opportunity to run a cycle -- that
rerun IS the scheduler here. This is a deliberate, known tradeoff for a
demo/competition context, not a substitute for serve.py's real daemon:
closing every tab silently stops trading with no alarm, and it inherits
whatever the current market-hours/interval state is only while a tab is
open. See README.md "Hosting" section for the full comparison.

Secrets required (Streamlit Cloud: Settings -> Secrets, or locally in
.streamlit/secrets.toml) -- same values as the project's own .env:
    ALPACA_API_KEY = "..."
    ALPACA_SECRET_KEY = "..."
    ALPACA_PAPER_TRADE = "true"
    FEATHERLESS_API_KEY = "..."
    FEATHERLESS_MODEL = "moonshotai/Kimi-K2-Instruct"   # optional
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="VRP Agent — Live Demo", page_icon="📈", layout="wide")

# ---------- restyle Streamlit's default widgets to match the custom dashboard ----------
st.markdown("""
<style>
  #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
  .block-container { padding-top: 1.5rem; max-width: 1400px; }

  h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; letter-spacing: 0.2px; }
  h1 { font-size: 1.5rem !important; }
  h2, h3 { font-size: 0.95rem !important; text-transform: uppercase; letter-spacing: 0.6px; color: #8b93a7 !important; }

  /* metric cards */
  div[data-testid="stMetric"] {
    background: #12161f; border: 1px solid #1f2531; border-radius: 10px;
    padding: 14px 16px 10px 16px;
  }
  div[data-testid="stMetricLabel"] { color: #8b93a7 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.5px; }
  div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
  div[data-testid="stMetricDelta"] svg { display: none; }

  /* progress bars */
  div[data-testid="stProgress"] > div > div { background: #1a1f2b !important; }
  div[data-testid="stProgress"] > div > div > div { background: linear-gradient(90deg, #2d9d70, #3ddc97) !important; }

  /* dataframes */
  div[data-testid="stDataFrame"] { border: 1px solid #1f2531; border-radius: 8px; overflow: hidden; }

  /* expanders (decision feed) */
  div[data-testid="stExpander"] {
    background: #0e121a; border: 1px solid #191d27 !important; border-radius: 7px;
  }

  /* toggles */
  div[data-testid="stToggle"] label p { font-size: 0.82rem; color: #8b93a7; }

  hr { border-color: #1f2531 !important; margin: 1.2rem 0 !important; }

  /* alerts/info boxes */
  div[data-testid="stAlert"] { border-radius: 8px; }

  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-thumb { background: #232a38; border-radius: 4px; }

  /* strategy step flow */
  .flow-row { display: flex; gap: 10px; align-items: stretch; margin: 6px 0 4px 0; }
  .flow-step {
    flex: 1; background: #12161f; border: 1px solid #1f2531; border-radius: 10px;
    padding: 14px 14px 12px 14px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
  }
  .flow-step .badge {
    display: inline-block; font-size: 9.5px; font-weight: 700; letter-spacing: 0.4px;
    padding: 2px 7px; border-radius: 4px; margin-bottom: 8px;
  }
  .badge-code { background: rgba(61,220,151,0.15); color: #3ddc97; }
  .badge-ai { background: rgba(124,140,255,0.18); color: #7c8cff; }
  .flow-step .step-title { font-size: 13.5px; font-weight: 700; color: #e6e9ef; margin-bottom: 4px; }
  .flow-step .step-desc { font-size: 11.5px; color: #8b93a7; line-height: 1.4; }
  .flow-arrow { display: flex; align-items: center; color: #3a4258; font-size: 18px; padding: 0 2px; }
  @media (max-width: 900px) { .flow-row { flex-direction: column; } .flow-arrow { display: none; } }
</style>
""", unsafe_allow_html=True)

# ---------- credentials: Streamlit secrets -> env vars, so config.load_config() finds them ----------
for _key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER_TRADE", "FEATHERLESS_API_KEY", "FEATHERLESS_MODEL"]:
    if _key in st.secrets and _key not in os.environ:
        os.environ[_key] = str(st.secrets[_key])
os.environ.setdefault("ALPACA_PAPER_TRADE", "true")
os.environ.setdefault("AGENT_DB_PATH", "./data/agent.db")

missing = [k for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "FEATHERLESS_API_KEY") if not os.environ.get(k)]
if missing:
    st.error(
        f"Missing required secrets: {', '.join(missing)}. "
        "Set them in Streamlit Cloud under Settings → Secrets (same values as your .env)."
    )
    st.stop()

import streamlit.components.v1 as components

from alpaca_quant_agent import control, dashboard, ledger, universe
from alpaca_quant_agent.config import load_config
from alpaca_quant_agent.cycle import run_one_cycle
from alpaca_quant_agent.execution.alpaca_mcp import AlpacaMcpClient
from alpaca_quant_agent.scheduler import market_is_open

config = load_config()
interval_seconds = int(config.get("scheduler", "cycle_interval_minutes", default=15)) * 60


@st.cache_data(ttl=60, show_spinner=False)
def fetch_real_account_state(_config):
    """Alpaca's own truth about the account -- equity, cash, and every
    currently-open position -- regardless of who or what opened them.
    Distinct from the ledger-based panels below, which only know about
    positions THIS bot's own code opened; an account with prior activity
    from another process (a teammate's daemon, manual trades) has real
    positions the ledger has never heard of. Cached briefly since this is a
    real network call (spins up the MCP subprocess) on every script rerun."""
    async def _fetch():
        async with AlpacaMcpClient(_config) as client:
            account = await client.get_account()
            positions = await client.get_positions()
            return account, positions
    return asyncio.run(_fetch())


def fmt_usd(n, show_plus: bool = False) -> str:
    if n is None:
        return "—"
    sign = "-" if n < 0 else ("+" if show_plus and n > 0 else "")
    return f"{sign}${abs(n):,.2f}"


def fmt_pct(n, show_plus: bool = False) -> str:
    if n is None:
        return "—"
    sign = "+" if show_plus and n > 0 else ""
    return f"{sign}{n * 100:.2f}%"


# ---------- header ----------
st.title("📈 VRP-Harvesting Options Agent — Live Demo")

st.markdown("""
<div class="flow-row">
  <div class="flow-step">
    <span class="badge badge-code">CODE</span>
    <div class="step-title">1. Screen</div>
    <div class="step-desc">IV rank + trend regime + a curated quality universe. Pure math, no AI.</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <span class="badge badge-code">CODE</span>
    <div class="step-title">2. Risk-Gate</div>
    <div class="step-desc">Every candidate must clear a full suite of hard, unit-tested risk limits before anything else sees it.</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <span class="badge badge-ai">AI · narrow</span>
    <div class="step-title">3. AI Review</div>
    <div class="step-desc">Only job: skim news, then pick from that pre-approved list. Can't invent trades, sizes, or strikes.</div>
  </div>
  <div class="flow-arrow">→</div>
  <div class="flow-step">
    <span class="badge badge-code">CODE</span>
    <div class="step-title">4. Execute</div>
    <div class="step-desc">Gates are re-checked server-side right before the order is ever submitted.</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("The AI never sees a blank canvas — it's picking from an already-safe menu, not deciding what's safe. "
           "97 unit tests cover the deterministic steps.")

with st.expander("ℹ️ How this works (read first if you're new here)", expanded=False):
    st.markdown(
        "- **This page runs the real trading pipeline itself**, on every auto-refresh — "
        "there's no separate always-on server. That means it only keeps \"trading\" for as "
        "long as at least one browser tab stays open on this URL.\n"
        "- **Trading Mode** below controls what actually happens each cycle (see the toggles).\n"
        "- **Real Account** tab = ground truth from Alpaca itself (equity, real positions) — "
        "shows real activity from *anyone/anything* that's touched this account, not just this bot.\n"
        "- **Bot Activity** tab = only what *this bot's own code* has done — will say \"no data yet\" "
        "until it actually completes a full cycle (which only happens while the market's open)."
    )

# ---------- trading mode: one clear status line + the two controls that drive it ----------
st.subheader("Trading Mode")
current_live = control.get_live_mode(config.db_path)
halt_state = control.get_halt_state(config.db_path)

if halt_state.halted:
    st.warning("⏸️ **PAUSED** — the bot will not open any new positions. Existing positions are still "
               "monitored and can still be closed.")
elif current_live:
    st.error("🔴 **LIVE** — this bot can and will place real orders on your Alpaca **paper** "
              "(simulated-money) account. No real cash is ever at risk, but real paper trades will appear "
              "in your Alpaca dashboard.")
else:
    st.info("🟡 **DRY RUN** — the bot runs its full decision process and logs exactly what it *would* "
            "do, but never actually submits an order. Safe default.")

mode_l, mode_r = st.columns(2)
with mode_l:
    new_live = st.toggle("Place real paper orders", value=current_live)
    st.caption("OFF (default): simulate only, nothing sent to Alpaca. "
               "ON: real orders go to your Alpaca **paper** account (fake money, real order flow).")
    if new_live != current_live:
        control.set_live_mode(config.db_path, new_live)
        st.rerun()
with mode_r:
    want_halt = st.toggle("Pause new entries", value=halt_state.halted)
    st.caption("Independent emergency stop: when ON, blocks opening *new* trades regardless of the "
               "toggle to the left, but still lets existing positions be managed/closed normally.")
    if want_halt != halt_state.halted:
        control.set_halt_state(config.db_path, want_halt, reason="toggled from Streamlit demo")
        st.rerun()

st.divider()

# ---------- the "scheduler": autorefresh forces a rerun every cycle_interval_minutes ----------
st_autorefresh(interval=interval_seconds * 1000, key="cycle_autorefresh")

last_run_path = Path(config.db_path).with_name("last_cycle_at.txt")
now = time.time()
last_run = float(last_run_path.read_text()) if last_run_path.exists() else 0.0
due = (now - last_run) >= (interval_seconds - 5)  # small slack for autorefresh jitter

status_box = st.empty()
if due:
    last_run_path.parent.mkdir(parents=True, exist_ok=True)
    last_run_path.write_text(str(now))
    with status_box, st.spinner("Running a cycle…"):
        try:
            if asyncio.run(market_is_open(config)):
                summary = asyncio.run(run_one_cycle(config, dry_run=not current_live))
                ledger.log_decision(config.db_path, candidate_id=None, symbol=None,
                                     decision="cycle_ran", detail=summary)
            else:
                summary = "Market closed — skipping this cycle."
                ledger.log_decision(config.db_path, candidate_id=None, symbol=None,
                                     decision="cycle_skipped", detail="market closed")
        except Exception as exc:  # noqa: BLE001 -- show it, don't crash the page
            summary = f"Cycle error: {exc}"
            ledger.log_decision(config.db_path, candidate_id=None, symbol=None,
                                 decision="cycle_error", detail=str(exc))
    st.session_state["last_summary"] = summary
    st.session_state["last_summary_at"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

if "last_summary" in st.session_state:
    st.caption(f"**Last cycle** ({st.session_state.get('last_summary_at', '—')}): {st.session_state['last_summary']}")
else:
    seconds_until_due = max(0, int(interval_seconds - (now - last_run)))
    st.caption(f"Next cycle in ~{seconds_until_due}s (or on the next page refresh after that).")

st.divider()

tab_chart, tab_account, tab_activity = st.tabs(["📊 Live Chart", "💰 Real Account (Alpaca)", "🤖 Bot Activity"])

with tab_chart:
    chart_symbol = st.selectbox("Symbol", options=list(universe.SYMBOLS),
                                 index=list(universe.SYMBOLS).index("SPY") if "SPY" in universe.SYMBOLS else 0)
    components.html(f"""
    <div class="tradingview-widget-container" style="height:420px;">
      <div id="tv_chart" style="height:100%;"></div>
    </div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
      autosize: true,
      symbol: "{chart_symbol}",
      interval: "15",
      timezone: "America/New_York",
      theme: "dark",
      style: "1",
      locale: "en",
      toolbar_bg: "#12161f",
      enable_publishing: false,
      studies: ["STD;EMA", "STD;ADX"],
      container_id: "tv_chart",
    }});
    </script>
    """, height=430)

with tab_account:
    st.caption("Equity, cash, and every open position exactly as Alpaca reports them right now — "
               "includes positions opened by anyone/anything, not just this bot.")
    try:
        real_account, real_positions = fetch_real_account_state(config)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Real Equity", fmt_usd(float(real_account.get("equity", 0))))
        rc2.metric("Cash", fmt_usd(float(real_account.get("cash", 0))))
        rc3.metric("Real Open Positions", len(real_positions))

        if real_positions:
            rp_df = pd.DataFrame(real_positions)
            cols = ["symbol", "side", "qty", "avg_entry_price", "current_price", "market_value", "unrealized_pl", "unrealized_plpc"]
            rp_df = rp_df[[c for c in cols if c in rp_df.columns]]
            for numeric_col in ["qty", "avg_entry_price", "current_price", "market_value", "unrealized_pl", "unrealized_plpc"]:
                if numeric_col in rp_df.columns:
                    rp_df[numeric_col] = pd.to_numeric(rp_df[numeric_col], errors="coerce")
            rp_df.columns = [c.replace("_", " ").title() for c in rp_df.columns]
            st.dataframe(rp_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No open positions on this account right now.")
    except Exception as exc:  # noqa: BLE001 -- show it, don't crash the page
        st.warning(f"Couldn't fetch real account state: {exc}")

with tab_activity:
    data = dashboard.build_snapshot()

    if not data.get("has_data"):
        st.info("No trading data yet — this bot hasn't completed a full cycle (only happens while "
                "the market's open). Check the **Real Account** tab for the account's actual current "
                "state in the meantime.")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Portfolio Equity", fmt_usd(data["equity"]))
        c2.metric("Total P&L", fmt_usd(data["total_pnl"], show_plus=True), fmt_pct(data["total_pnl_pct"], show_plus=True))
        c3.metric("Today's P&L", fmt_pct(data["today_pnl_pct"], show_plus=True))
        c4.metric("Drawdown from Peak", fmt_pct(data["drawdown_pct"]))
        c5.metric("Open Positions", f"{data['open_position_count']} / {data['risk_limits']['max_total_positions']}")
        cs = data.get("closed_stats") or {}
        c6.metric("Win Rate (closed)", fmt_pct(cs.get("win_rate")) if cs.get("win_rate") is not None else "—",
                  f"{cs.get('count', 0)} closed" if cs.get("count") else None)

        left, right = st.columns([1.6, 1])
        with left:
            st.subheader("Equity Curve")
            curve = data.get("equity_curve") or []
            if curve:
                df = pd.DataFrame(curve)
                df["date"] = pd.to_datetime(df["date"])
                st.line_chart(df.set_index("date")["equity"], height=280)
            else:
                st.caption("No equity history yet.")

        with right:
            st.subheader("Risk Gate Utilization")
            rl = data["risk_limits"]

            def gate_bar(label: str, value: float, cap) -> None:
                pct = min(abs(value) / cap, 1.0) if cap else 0.0
                st.caption(f"{label} — {fmt_pct(value)} / cap {fmt_pct(cap) if cap is not None else '—'}")
                st.progress(pct)

            gate_bar("Portfolio Heat", data["portfolio_heat_pct"], rl["max_portfolio_heat_pct"])
            gate_bar("Net Delta / Equity", data["net_delta_pct"], rl["portfolio_delta_band_pct"])
            gate_bar("Net Vega / Equity", data["net_vega_pct"], rl["portfolio_vega_cap_pct"])
            gate_bar("Sleeve B Allocation", data["sleeve_b_heat_pct"], rl["sleeve_b_max_allocation_pct"])

        st.divider()

        left, right = st.columns([1.6, 1])
        with left:
            st.subheader(f"Open Positions ({len(data['open_positions'])})")
            if data["open_positions"]:
                df = pd.DataFrame(data["open_positions"])
                df = df[["symbol", "sleeve", "strategy_type", "contracts", "credit_received",
                          "max_loss", "net_delta", "net_vega", "dte", "days_to_earnings"]]
                df.columns = ["Symbol", "Sleeve", "Strategy", "Contracts", "Credit",
                              "Max Loss", "Δ", "Vega", "DTE", "Days to Earnings"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No open positions.")

        with right:
            st.subheader("Correlation Bucket Heat")
            buckets = data.get("bucket_heat") or []
            if buckets:
                for b in buckets:
                    cap = b.get("cap_pct")
                    pct = min(b["pct"] / cap, 1.0) if cap else 0.0
                    st.caption(f"{b['bucket'].replace('_', ' ')} — {fmt_pct(b['pct'])} / {fmt_pct(cap) if cap else '—'}")
                    st.progress(pct)
            else:
                st.caption("No open positions in any correlation bucket.")

        st.divider()

        left, right = st.columns([1.6, 1])
        with left:
            st.subheader(f"Recent Trades ({len(data['recent_trades'])})")
            if data["recent_trades"]:
                df = pd.DataFrame(data["recent_trades"])
                df = df[["created_at", "symbol", "sleeve", "action", "contracts", "credit_or_debit", "rationale"]]
                df.columns = ["Time", "Symbol", "Sleeve", "Action", "Contracts", "Credit/Debit", "Rationale"]
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No trades yet.")

        with right:
            st.subheader("Decision Feed")
            decisions = data.get("recent_decisions") or []
            if decisions:
                for d in decisions[:20]:
                    label = f"{d['symbol']} — {d['decision'].replace('_', ' ')}" if d.get("symbol") else d["decision"].replace("_", " ")
                    with st.expander(label, expanded=False):
                        st.caption(d.get("created_at", ""))
                        st.write(d.get("detail") or "—")
            else:
                st.caption("No decisions logged yet.")

    st.divider()
    with st.expander("Cycle history (every check-in, not just trades)", expanded=False):
        rows = ledger.recent_cycle_log(config.db_path, limit=30)
        if rows:
            hist = pd.DataFrame(rows)
            hist.columns = ["Time", "Result", "Detail"]
            st.dataframe(hist, use_container_width=True, hide_index=True)
        else:
            st.caption("No cycle attempts logged yet.")
