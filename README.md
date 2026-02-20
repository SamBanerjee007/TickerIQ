# TickerIQ — Intelligent Ticker Scoring

A public stock analysis tool that aggregates technical, fundamental, sentiment,
and relative-performance signals into a single weighted score (1–5) and
generates a **Buy / Hold / Sell** signal — tuned to your trading style.

> **Not financial advice.** Data sourced from Yahoo Finance. Always do your own research.

---

## Features

| Category | Metrics |
|---|---|
| **Technicals** | RSI, MACD, Bollinger Bands, Volume trend, ADX trend strength |
| **Support & Resistance** | Pivot Points (PP/R1/R2/R3/S1/S2/S3) + 52-week H/L + swing highs/lows |
| **Sentiment** | Put/Call ratio, Options open interest, Short float % |
| **Fundamentals** | P/E (TTM + Forward), P/B, PEG, EPS growth, Revenue growth, Dividend yield, D/E |
| **Relative Performance** | vs SPY, vs Sector ETF |
| **Analyst Consensus** | Avg rating + price target upside |

**Trading style profiles** adjust metric weights:
- **Swing Trader** — technicals and sentiment dominate
- **Position Trader** — balanced mix
- **Long-term Investor** — fundamentals and growth dominate

---

## Quick Start (local)

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/tickeriq.git
cd tickeriq

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml — add your Supabase + Turnstile keys

# 4. Run
streamlit run app.py
```

The app works without Supabase (no usage logging) and without a real
Turnstile key (uses the test key that always passes).

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select this repo → `app.py`.
3. In **Advanced settings → Secrets**, paste the contents of your `secrets.toml`.
4. Click **Deploy**.

---

## Supabase Setup

1. Create a free project at [supabase.com](https://supabase.com).
2. Go to **SQL Editor** and run the contents of [`supabase_setup.sql`](supabase_setup.sql).
3. Copy your **Project URL** and **anon public key** from Project Settings → API
   into `secrets.toml`.

---

## Cloudflare Turnstile Setup

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com) → **Turnstile** → Add site.
2. Choose **Invisible** widget type.
3. Add your Streamlit app domain (e.g. `yourapp.streamlit.app`).
4. Copy **Site Key** and **Secret Key** into `secrets.toml`.

> For local development, use site key `1x00000000000000000000AA` and
> secret key `1x0000000000000000000000000000000AA` (Cloudflare test keys).

---

## Scoring Logic

Each metric is independently scored 1–5 (bearish → bullish), then combined
via a weighted average. The final score maps to a signal:

| Score | Signal |
|---|---|
| 4.0 – 5.0 | Strong Buy |
| 3.25 – 4.0 | Buy |
| 2.75 – 3.25 | Hold |
| 2.0 – 2.75 | Sell |
| 1.0 – 2.0 | Strong Sell |

Full weight tables and per-metric scoring logic are in [`scoring.py`](scoring.py).

---

## Project Structure

```
tickeriq/
├── app.py                   # Streamlit UI — all three pages
├── stock_metrics.py         # Data fetching (yfinance + pandas-ta)
├── support_resistance.py    # Pivot points + rolling S/R
├── scoring.py               # Weighted scoring engine
├── db.py                    # Supabase: logging, trending, admin stats
├── requirements.txt
├── supabase_setup.sql       # Run once to create the DB table
├── .streamlit/
│   ├── config.toml          # Dark theme + server settings
│   └── secrets.toml.example # Template — copy to secrets.toml
└── .gitignore               # Excludes secrets.toml
```

---

## Contributing

PRs are welcome! Please open an issue first to discuss major changes.

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes
4. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT — see [LICENSE](LICENSE).
