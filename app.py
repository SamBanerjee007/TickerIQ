"""
TickerIQ — Intelligent Ticker Scoring
Streamlit public web application.

Pages:
  Main   — Enter ticker, pick trading style, get scored analysis + signal
  Admin  — Password-protected usage dashboard (access via ?admin=1)
"""

import math
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from stock_metrics      import get_all_metrics, SECTOR_ETF_MAP
from support_resistance import get_sr_context
from scoring            import calculate_score, _METRIC_LABELS
from db                 import log_query, check_rate_limit, get_trending, get_admin_stats

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TickerIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Hide sidebar and its toggle button */
  [data-testid="stSidebar"]        { display: none !important; }
  [data-testid="collapsedControl"] { display: none !important; }

  /* Reduce default Streamlit side padding so content uses the full width */
  .block-container {
    padding-left:  2rem !important;
    padding-right: 2rem !important;
    padding-top:   1.5rem !important;
    max-width: 100% !important;
  }

  /* Plotly charts: never clip overflow */
  [data-testid="stPlotlyChart"] { overflow: visible !important; }

  .main-header { font-size: 2.4rem; font-weight: 800; color: #00D4AA; margin-bottom: 0; }
  .sub-header  { font-size: 1rem; color: #888; margin-top: 0; margin-bottom: 1.5rem; }
  .signal-badge {
    display: inline-block;
    padding: 0.35rem 1.2rem;
    border-radius: 20px;
    font-size: 1.1rem;
    font-weight: 700;
    color: #0E1117;
  }
  .metric-row { border-left: 3px solid #333; padding-left: 0.6rem; margin-bottom: 0.4rem; }
  .score-num  { font-size: 3rem; font-weight: 800; }
  .trend-card {
    background: #1A1D27;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
  }
  .sr-level  { font-family: monospace; font-size: 0.9rem; }
  hr.divider { border-color: #333; margin: 1.5rem 0; }

  /* Responsive: stack the top-3 columns on narrow viewports */
  @media (max-width: 900px) {
    [data-testid="stHorizontalBlock"] > div { min-width: 100% !important; }
  }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────

if "results"             not in st.session_state: st.session_state.results = []
if "captcha_verified"    not in st.session_state: st.session_state.captcha_verified = False
if "session_query_count" not in st.session_state: st.session_state.session_query_count = 0
if "admin_auth"          not in st.session_state: st.session_state.admin_auth = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fmt(v, suffix="", decimals=2, na="N/A"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return na
    return f"{round(v, decimals)}{suffix}"


def _pct(v, na="N/A"):
    if v is None:
        return na
    return f"{round(v * 100, 2)}%"


def _fmt_large(v):
    if v is None:
        return "N/A"
    v = float(v)
    if v >= 1e12:  return f"${v/1e12:.2f}T"
    if v >= 1e9:   return f"${v/1e9:.2f}B"
    if v >= 1e6:   return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"


def _score_color(s: int) -> str:
    colors = {1: "#D50000", 2: "#FF6D00", 3: "#FFD600", 4: "#69F0AE", 5: "#00C853"}
    return colors.get(s, "#888")


def _signal_emoji(sig: str) -> str:
    return {"Strong Buy": "🟢", "Buy": "🟩", "Hold": "🟡", "Sell": "🟧", "Strong Sell": "🔴"}.get(sig, "")


# ── Cloudflare Turnstile verification ────────────────────────────────────────

def _verify_turnstile(token: str) -> bool:
    """Server-side verification of Turnstile token."""
    try:
        try:
            secret = st.secrets["turnstile"]["secret_key"]
        except Exception:
            secret = ""
        if not secret:
            return True  # skip if not configured
        resp = requests.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={"secret": secret, "response": token},
            timeout=5,
        )
        return resp.json().get("success", False)
    except Exception:
        return True


def _has_real_turnstile() -> bool:
    """True only when a real (non-test) Turnstile secret key is configured."""
    try:
        key = st.secrets["turnstile"]["secret_key"]
        return bool(key) and not key.startswith("1x000000")
    except Exception:
        return False


def _render_turnstile() -> None:
    try:
        site_key = st.secrets["turnstile"]["site_key"]
    except Exception:
        site_key = "1x00000000000000000000AA"
    import streamlit.components.v1 as components
    components.html(f"""
        <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
        <div class="cf-turnstile"
             data-sitekey="{site_key}"
             data-theme="dark"
             data-callback="onTurnstileSuccess">
        </div>
        <script>
          function onTurnstileSuccess(token) {{
            try {{
              window.parent.sessionStorage.setItem('turnstile_token', token);
            }} catch(e) {{
              sessionStorage.setItem('turnstile_token', token);
            }}
          }}
        </script>
    """, height=70, scrolling=False)


def _get_captcha_token() -> str | None:
    try:
        from streamlit_js_eval import streamlit_js_eval
        return streamlit_js_eval(
            js_expressions="sessionStorage.getItem('turnstile_token')",
            key="read_turnstile_token",
        )
    except Exception:
        return None


# ── Score gauge chart ─────────────────────────────────────────────────────────

def _gauge(score: float, signal: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 40, "color": color}, "suffix": ""},
        gauge={
            "axis": {"range": [1, 5], "tickwidth": 1, "tickcolor": "#555",
                     "tickvals": [1, 2, 3, 4, 5],
                     "ticktext": ["1", "2", "3", "4", "5"]},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1A1D27",
            "borderwidth": 0,
            "steps": [
                {"range": [1.0, 2.0],  "color": "#2a0a0a"},
                {"range": [2.0, 2.75], "color": "#2a1500"},
                {"range": [2.75, 3.25],"color": "#2a2500"},
                {"range": [3.25, 4.0], "color": "#0a2a15"},
                {"range": [4.0, 5.0],  "color": "#0a2a1a"},
            ],
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.8, "value": score},
        },
        title={"text": signal, "font": {"size": 20, "color": color}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=55, b=10),
        paper_bgcolor="#0E1117",
        font={"color": "#FAFAFA"},
    )
    return fig


# ── Component score bar chart ─────────────────────────────────────────────────

def _component_chart(component_scores: dict) -> go.Figure:
    labels  = [v["label"]  for v in component_scores.values()]
    scores  = [v["score"]  for v in component_scores.values()]
    weights = [v["weight"] for v in component_scores.values()]
    colors  = [_score_color(s) for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s}/5 ({w*100:.0f}%)" for s, w in zip(scores, weights)],
        textposition="outside",
        hovertemplate="%{y}<br>Score: %{x}/5<extra></extra>",
    ))
    fig.update_layout(
        xaxis={"range": [0, 6], "tickvals": [1, 2, 3, 4, 5]},
        height=max(280, len(labels) * 28),
        margin=dict(l=10, r=80, t=10, b=10),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={"color": "#FAFAFA", "size": 11},
        showlegend=False,
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis Page
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _render_sr(sr_data: dict, current_price: float | None) -> None:
    pivots  = sr_data.get("pivots",  {})
    rolling = sr_data.get("rolling", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Pivot Points** *(prior day OHLC)*")
        if pivots:
            levels = [
                ("R3", pivots.get("R3"), "#D50000"),
                ("R2", pivots.get("R2"), "#FF6D00"),
                ("R1", pivots.get("R1"), "#FFD600"),
                ("PP", pivots.get("PP"), "#888"),
                ("S1", pivots.get("S1"), "#69F0AE"),
                ("S2", pivots.get("S2"), "#00C853"),
                ("S3", pivots.get("S3"), "#00796B"),
            ]
            for label, val, color in levels:
                if val is None:
                    continue
                marker = " ◀ current" if current_price and abs(val - current_price) < current_price * 0.005 else ""
                st.markdown(
                    f'<div class="sr-level" style="color:{color}">'
                    f'  {label}: <b>${val:,.2f}</b>{marker}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Pivot data unavailable")

    with col2:
        st.markdown("**Rolling Levels**")
        if rolling:
            st.markdown(f"52-week High: **${rolling.get('52w_high', 'N/A'):,.2f}**")
            st.markdown(f"52-week Low:  **${rolling.get('52w_low',  'N/A'):,.2f}**")
            st.markdown(f"20-day High:  **${rolling.get('20d_high', 'N/A'):,.2f}**")
            st.markdown(f"20-day Low:   **${rolling.get('20d_low',  'N/A'):,.2f}**")

            if rolling.get("swing_highs"):
                st.markdown("Swing Highs: " + "  ·  ".join(
                    f"**${v:,.2f}**" for v in rolling["swing_highs"]
                ))
            if rolling.get("swing_lows"):
                st.markdown("Swing Lows: " + "  ·  ".join(
                    f"**${v:,.2f}**" for v in rolling["swing_lows"]
                ))
        else:
            st.caption("Rolling S/R data unavailable")


def _render_result(result: dict) -> None:
    metrics = result["metrics"]
    sr_data = result["sr_data"]
    scored  = result["scored"]

    tech  = metrics.get("technicals",  {})
    fund  = metrics.get("fundamentals", {})
    opts  = metrics.get("options",      {})
    perf  = metrics.get("performance",  {})

    symbol  = metrics["symbol"]
    score   = scored["total_score"]
    signal  = scored["signal"]
    color   = scored["signal_color"]
    current = tech.get("current_price") or fund.get("current_price")
    period  = perf.get("period", "?")
    period_label = {"3mo": "3 months", "6mo": "6 months", "1y": "1 year"}.get(period, period)

    # ── Header row ────────────────────────────────────────────────────────────
    st.markdown(f"## {symbol} — {fund.get('name', symbol)}")
    st.caption(f"{fund.get('sector','?')} · {fund.get('industry','?')} · "
               f"Style: {metrics['trading_style']} · Lookback: {period_label}")

    top_col1, top_col2, top_col3 = st.columns([1, 1.8, 1.8])

    with top_col1:
        st.plotly_chart(_gauge(score, signal, color), width="stretch")

    with top_col2:
        st.markdown("#### Market Snapshot")
        mc = _fmt_large(fund.get("market_cap"))
        st.markdown(f"| | |\n|---|---|\n"
                    f"| **Price**    | ${_fmt(current)} |\n"
                    f"| **Mkt Cap**  | {mc} |\n"
                    f"| **52w High** | ${_fmt(fund.get('52w_high'))} |\n"
                    f"| **52w Low**  | ${_fmt(fund.get('52w_low'))} |\n"
                    f"| **Beta**     | {_fmt(fund.get('beta'))} |")

    with top_col3:
        st.markdown(f"#### vs Market *(last {period_label})*")
        sr  = perf.get("stock_return")
        spy = perf.get("spy_return")
        sec = perf.get("sector_return")
        etf = metrics.get("sector_etf", "—")
        st.markdown(f"| | |\n|---|---|\n"
                    f"| **{symbol}** | {_fmt(sr, '%')} |\n"
                    f"| **SPY** | {_fmt(spy, '%')} |\n"
                    f"| **{etf}** | {_fmt(sec, '%')} |")
        if sr is not None and spy is not None:
            diff = sr - spy
            arrow = "▲" if diff >= 0 else "▼"
            color_txt = "green" if diff >= 0 else "red"
            st.markdown(f"vs SPY: <span style='color:{color_txt}'>{arrow} {abs(diff):.2f}%</span>",
                        unsafe_allow_html=True)

    st.divider()

    # ── Detail tabs ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Technicals", "📍 Support & Resistance",
        "📊 Fundamentals", "🎯 Options & Sentiment", "⚖️ Score Breakdown",
    ])

    with tab1:
        c1, c2, c3 = st.columns(3)
        rsi = tech.get("rsi")
        with c1:
            st.metric("RSI (14-day)", _fmt(rsi),
                      delta=f"Trend {tech.get('rsi_trend','?')}",
                      delta_color="normal")
            if rsi:
                zone = ("Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral")
                st.caption(f"Zone: **{zone}**")

        macd = tech.get("macd"); msig = tech.get("macd_signal")
        with c2:
            st.metric("MACD vs Signal",
                      f"{_fmt(macd, decimals=4)} / {_fmt(msig, decimals=4)}",
                      delta=tech.get("macd_trend", ""),
                      delta_color="normal")
            st.caption(f"Histogram: {_fmt(tech.get('macd_hist'), decimals=4)}")

        with c3:
            st.metric("ADX (Trend Strength)", _fmt(tech.get("adx")))
            st.caption("≥25 trending · ≥40 strong")

        c4, c5, c6 = st.columns(3)
        with c4:
            bp = tech.get("bb_pct")
            st.metric("Bollinger %B", f"{_fmt(bp)}%")
            st.caption(f"Lower: ${_fmt(tech.get('bb_lower'))} · "
                       f"Upper: ${_fmt(tech.get('bb_upper'))}")
        with c5:
            vr = tech.get("volume_ratio")
            st.metric("Volume vs 20d Avg", f"{_fmt(vr)}×",
                      delta="High" if vr and vr > 1.5 else "Normal",
                      delta_color="normal" if vr and vr > 1.5 else "off")
            st.caption(f"Today: {_fmt(tech.get('volume_current'), decimals=0)} · "
                       f"Avg: {_fmt(tech.get('volume_20d_avg'), decimals=0)}")
        with c6:
            st.metric("5-day Trend", tech.get("trend", "N/A"))

    with tab2:
        _render_sr(sr_data, current)

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Valuation**")
            st.metric("P/E (TTM)",    _fmt(fund.get("pe_trailing")))
            st.metric("Forward P/E",  _fmt(fund.get("pe_forward")))
            st.metric("P/B Ratio",    _fmt(fund.get("pb_ratio")))
            st.metric("PEG Ratio",    _fmt(fund.get("peg_ratio")))
        with c2:
            st.markdown("**Growth**")
            st.metric("EPS Growth QoQ",    _pct(fund.get("eps_growth_qoq")))
            st.metric("Revenue Growth YoY", _pct(fund.get("revenue_growth_yoy")))
            st.metric("Dividend Yield",     _pct(fund.get("dividend_yield")))
        with c3:
            st.markdown("**Balance Sheet & Analyst**")
            st.metric("Debt / Equity",  _fmt(fund.get("debt_to_equity")))
            st.metric("Short Float",    _pct(fund.get("short_float")))

            rating     = fund.get("analyst_rating", "N/A")
            target     = fund.get("target_mean_price")
            n_analysts = fund.get("num_analyst_opinions")
            st.metric("Analyst Rating", str(rating).title() if rating else "N/A")
            if target and current:
                upside = (target - current) / current * 100
                st.metric("Avg Price Target", f"${_fmt(target)}",
                          delta=f"{upside:+.1f}% upside",
                          delta_color="normal" if upside >= 0 else "inverse")
            if n_analysts:
                st.caption(f"Based on {n_analysts} analyst opinions")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            pcr  = opts.get("pc_ratio")
            sent = opts.get("sentiment", "N/A")
            st.metric("Put/Call Ratio", _fmt(pcr) if pcr else "N/A")
            st.metric("Options Sentiment", sent)
            st.caption(f"Calls OI: {opts.get('calls_oi', 0):,} · "
                       f"Puts OI: {opts.get('puts_oi', 0):,}")
        with c2:
            short = fund.get("short_float")
            st.metric("Short Float %", _pct(short))
            if short is not None:
                pct = short * 100
                if pct > 20:
                    st.warning("High short interest — possible short squeeze or strong bearish conviction.")
                elif pct < 3:
                    st.success("Low short interest — few bears in this name.")

    with tab5:
        st.plotly_chart(_component_chart(scored["component_scores"]),
                        width="stretch")
        st.caption("Bar length = score (1–5). % label = weight in final score.")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)


def show_analysis_page():
    st.markdown('<h1 class="main-header">TickerIQ</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Intelligent stock scoring · '
        'Enter a ticker, pick your trading style, get a data-driven signal.</p>',
        unsafe_allow_html=True,
    )

    # ── Trending strip (only shows if Supabase is configured) ─────────────────
    trending = get_trending(hours=24, limit=5)
    if trending:
        cols = st.columns(len(trending) + 1)
        cols[0].markdown("**🔥 Trending**")
        for i, t in enumerate(trending):
            sig = t.get("top_signal", "Hold")
            emoji = _signal_emoji(sig)
            cols[i + 1].markdown(
                f'<div class="trend-card">'
                f'<b>{t["symbol"]}</b><br>'
                f'<small>{emoji} {sig} · {t["count"]}×</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.divider()

    # ── Turnstile widget (only when real keys are configured) ─────────────────
    if _has_real_turnstile():
        _render_turnstile()
        st.caption("Complete the security check above, then click Analyze.")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("analysis_form"):
        col_ticker, col_style = st.columns([2, 1])
        with col_ticker:
            raw_input = st.text_input(
                "Ticker symbol",
                placeholder="AAPL",
                help="Enter a single US stock ticker symbol (e.g. AAPL, MSFT, NVDA).",
            )
        with col_style:
            trading_style = st.selectbox(
                "Trading style",
                ["Swing Trader", "Position Trader", "Long-term Investor"],
                index=1,
                help=(
                    "Swing Trader: 3-month window, technicals-heavy.\n"
                    "Position Trader: 6-month window, balanced mix.\n"
                    "Long-term Investor: 1-year window, fundamentals-heavy."
                ),
            )

        submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

    # ── On submit ─────────────────────────────────────────────────────────────
    if submitted:
        symbol = raw_input.strip().upper()
        if not symbol:
            st.warning("Please enter a ticker symbol.")
            return

        # CAPTCHA check — only enforced when real Turnstile keys are configured
        if _has_real_turnstile() and not st.session_state.captcha_verified:
            token = _get_captcha_token()
            if token:
                if _verify_turnstile(token):
                    st.session_state.captcha_verified = True
                else:
                    st.error("CAPTCHA verification failed. Please try again.")
                    return
            else:
                st.info("Please complete the security check above and click Analyze again.")
                return

        # Session rate limit
        if st.session_state.session_query_count >= 50:
            st.error("Session limit reached. Please refresh the page to continue.")
            return

        # IP rate limit
        if not check_rate_limit(max_per_hour=20):
            st.error("Rate limit exceeded (20 analyses/hour). Please wait before trying again.")
            return

        with st.spinner(f"Analyzing {symbol}…"):
            try:
                metrics = get_all_metrics(symbol, trading_style)
                sr_data = get_sr_context(symbol)
                scored  = calculate_score(metrics, sr_data)

                log_query(symbol, trading_style,
                          scored["total_score"], scored["signal"])
                st.session_state.session_query_count += 1

                st.session_state.results = [{
                    "metrics": metrics,
                    "sr_data": sr_data,
                    "scored":  scored,
                }]
            except Exception as e:
                st.error(f"Could not analyze **{symbol}**: {e}")
                st.session_state.results = []

    # ── Render results ────────────────────────────────────────────────────────
    for r in st.session_state.results:
        _render_result(r)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.caption("Data via Yahoo Finance · Not financial advice")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Admin Page
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def show_admin_page():
    st.header("🔐 Admin Dashboard")

    if not st.session_state.admin_auth:
        pwd = st.text_input("Password", type="password", key="admin_pwd")
        if st.button("Login"):
            try:
                correct = st.secrets["admin"]["password"]
            except Exception:
                correct = ""
            if pwd == correct:
                st.session_state.admin_auth = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    stats = get_admin_stats()
    if not stats:
        st.warning("No data available or database connection failed.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("All-time Queries", f"{stats['total']:,}")
    k2.metric("Today",            f"{stats['today']:,}")
    k3.metric("This Week",        f"{stats['this_week']:,}")
    k4.metric("This Month",       f"{stats['this_month']:,}")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)

    with ch1:
        st.subheader("Daily Query Volume")
        dv = stats.get("daily_volume")
        if dv is not None and not dv.empty:
            fig = px.bar(dv, x="date", y="queries",
                         color_discrete_sequence=["#00D4AA"])
            fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                              font={"color": "#FAFAFA"}, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch")

    with ch2:
        st.subheader("Signal Distribution")
        bs = stats.get("by_signal")
        if bs is not None and not bs.empty:
            fig = px.pie(bs, names=bs.columns[0], values=bs.columns[1],
                         color_discrete_sequence=px.colors.sequential.Teal)
            fig.update_layout(paper_bgcolor="#0E1117", font={"color": "#FAFAFA"},
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch")

    ch3, ch4 = st.columns(2)

    with ch3:
        st.subheader("Top Queried Symbols")
        ts = stats.get("top_symbols")
        if ts is not None and not ts.empty:
            fig = px.bar(ts.head(15), x=ts.columns[1], y=ts.columns[0],
                         orientation="h", color_discrete_sequence=["#69F0AE"])
            fig.update_layout(yaxis={"autorange": "reversed"},
                              paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                              font={"color": "#FAFAFA"}, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch")

    with ch4:
        st.subheader("Queries by Trading Style")
        by = stats.get("by_style")
        if by is not None and not by.empty:
            fig = px.pie(by, names=by.columns[0], values=by.columns[1],
                         color_discrete_sequence=px.colors.sequential.Teal_r)
            fig.update_layout(paper_bgcolor="#0E1117", font={"color": "#FAFAFA"},
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch")

    # ── Raw data ──────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Raw query log"):
        raw = stats.get("raw")
        if raw is not None:
            st.dataframe(
                raw[["created_at", "symbol", "trading_style", "score", "signal"]]
                  .sort_values("created_at", ascending=False)
                  .reset_index(drop=True),
                width="stretch",
            )

    if st.button("Logout"):
        st.session_state.admin_auth = False
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Router — URL-param based, no sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.query_params.get("admin") == "1":
    show_admin_page()
else:
    show_analysis_page()
