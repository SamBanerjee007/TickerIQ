"""
TickerIQ — Stock metrics collection module.
All market data sourced from Yahoo Finance via yfinance.
Technical indicators computed with pandas-ta (pure Python, no binary deps).
"""

import warnings
import datetime

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*utcnow.*")   # yfinance uses deprecated Timestamp.utcnow
# yfinance 0.2.50+ uses curl_cffi internally for browser impersonation.
# Do NOT pass a custom requests.Session — it will be rejected.

# ── Constants ─────────────────────────────────────────────────────────────────

SECTOR_ETF_MAP = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Healthcare":             "XLV",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
}

STYLE_PERIOD = {
    "Swing Trader":      "3mo",
    "Position Trader":   "6mo",
    "Long-term Investor": "1y",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(value, decimals=2):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return round(float(value), decimals)
    except Exception:
        return None


def _download(symbol: str, period: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Performance ───────────────────────────────────────────────────────────────

def get_stock_return(symbol: str, period: str = "6mo") -> float | None:
    try:
        df = _download(symbol, period=period)
        if df.empty or "Close" not in df.columns or len(df) < 2:
            return None
        start = _safe_float(df["Close"].iloc[0])
        end   = _safe_float(df["Close"].iloc[-1])
        if start is None or end is None or start == 0:
            return None
        return round(((end - start) / start) * 100, 2)
    except Exception:
        return None


# ── Technical indicators ──────────────────────────────────────────────────────

def get_technical_indicators(symbol: str, period: str = "6mo") -> dict:
    try:
        df = _download(symbol, period=period)
        if df.empty or len(df) < 35:
            return {}

        close  = df["Close"].astype(float)
        volume = df["Volume"].astype(float)

        # RSI
        rsi_series = ta.rsi(close, length=14)
        rsi      = _safe_float(rsi_series.iloc[-1])
        rsi_prev = _safe_float(rsi_series.iloc[-2])

        # MACD
        macd_df    = ta.macd(close, fast=12, slow=26, signal=9)
        macd_col   = [c for c in macd_df.columns if c.startswith("MACD_")][0]
        sig_col    = [c for c in macd_df.columns if c.startswith("MACDs_")][0]
        hist_col   = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
        macd_val   = _safe_float(macd_df[macd_col].iloc[-1], 4)
        macd_sig   = _safe_float(macd_df[sig_col].iloc[-1], 4)
        macd_hist  = _safe_float(macd_df[hist_col].iloc[-1], 4)

        # Bollinger Bands
        bb_df      = ta.bbands(close, length=20, std=2)
        bb_lower   = _safe_float(bb_df[[c for c in bb_df.columns if "BBL_" in c][0]].iloc[-1])
        bb_mid     = _safe_float(bb_df[[c for c in bb_df.columns if "BBM_" in c][0]].iloc[-1])
        bb_upper   = _safe_float(bb_df[[c for c in bb_df.columns if "BBU_" in c][0]].iloc[-1])
        current    = _safe_float(close.iloc[-1])
        bb_pct     = round((current - bb_lower) / (bb_upper - bb_lower) * 100, 1) \
                     if bb_upper and bb_lower and (bb_upper - bb_lower) > 0 else 50.0

        # ADX
        adx_df  = ta.adx(df["High"].astype(float), df["Low"].astype(float), close, length=14)
        adx_col = [c for c in adx_df.columns if c.startswith("ADX_")][0]
        adx     = _safe_float(adx_df[adx_col].iloc[-1])

        # Volume trend
        vol_20d_avg  = _safe_float(volume.iloc[-20:].mean(), 0)
        vol_current  = _safe_float(volume.iloc[-1], 0)
        vol_ratio    = round(vol_current / vol_20d_avg, 2) \
                       if vol_20d_avg and vol_20d_avg > 0 else 1.0

        # 5-day price trend
        recent = close.iloc[-5:].values
        if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
            trend = "Rising"
        elif all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
            trend = "Falling"
        else:
            trend = "Neutral"

        return {
            "current_price":  current,
            "rsi":            rsi,
            "rsi_prev":       rsi_prev,
            "rsi_trend":      "Up" if (rsi and rsi_prev and rsi > rsi_prev) else "Down",
            "macd":           macd_val,
            "macd_signal":    macd_sig,
            "macd_hist":      macd_hist,
            "macd_trend":     "Bullish" if (macd_val and macd_sig and macd_val > macd_sig) else "Bearish",
            "bb_upper":       bb_upper,
            "bb_mid":         bb_mid,
            "bb_lower":       bb_lower,
            "bb_pct":         bb_pct,
            "adx":            adx,
            "volume_current": vol_current,
            "volume_20d_avg": vol_20d_avg,
            "volume_ratio":   vol_ratio,
            "trend":          trend,
        }
    except Exception as e:
        return {}


# ── Options sentiment ─────────────────────────────────────────────────────────

def get_options_sentiment(symbol: str) -> dict:
    try:
        stock = yf.Ticker(symbol)
        expiries = stock.options  # tuple of expiry date strings; empty if blocked
        print(f"[TickerIQ] options {symbol}: expiries={expiries!r}")
        if not expiries:
            return {"pc_ratio": None, "sentiment": "N/A", "calls_oi": 0, "puts_oi": 0}

        today   = datetime.date.today()
        cutoff  = today + datetime.timedelta(days=90)
        calls_oi = puts_oi = 0

        dates = [d for d in expiries
                 if today <= datetime.datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]

        for d in dates:
            try:
                chain     = stock.option_chain(d)
                calls_oi += int(chain.calls["openInterest"].sum())
                puts_oi  += int(chain.puts["openInterest"].sum())
            except Exception as e:
                print(f"[TickerIQ] option_chain {symbol} {d}: {type(e).__name__}: {e}")
                continue

        pc_ratio = round(puts_oi / calls_oi, 2) if calls_oi > 0 else None

        if pc_ratio is None:
            sentiment = "N/A"
        elif pc_ratio < 0.7:
            sentiment = "Bullish"
        elif pc_ratio > 1.0:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"

        return {
            "pc_ratio":  pc_ratio,
            "sentiment": sentiment,
            "calls_oi":  calls_oi,
            "puts_oi":   puts_oi,
        }
    except Exception as e:
        print(f"[TickerIQ] options {symbol}: {type(e).__name__}: {e}")
        return {"pc_ratio": None, "sentiment": "N/A", "calls_oi": 0, "puts_oi": 0}


# ── Fundamentals ──────────────────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """
    Multi-tier fetch — each tier is independent; failures are logged + silenced.

    Tier 0   yf.download(period="1y")  — OHLC. Always works. Gives price,
             52w H/L, and computes beta vs SPY from daily returns.
    Tier 1   fast_info                  — market cap, currency (curl_cffi).
             yfinance 1.x renamed attrs: year_high/year_low (not fifty_two_week_*).
    Tier 2   .info                      — sector, P/E, analyst ratings, etc.
             Often blocked on cloud IPs; errors are printed to logs.
    Tier 3   yf.Search()               — lightweight search API fallback for
             sector/industry/name when Tier 2 fails on cloud.
    """
    result = {
        "symbol": symbol, "name": symbol,
        "sector": "Unknown", "industry": "Unknown",
        "market_cap": None, "currency": "USD", "current_price": None,
        "pe_trailing": None, "pe_forward": None, "pb_ratio": None,
        "peg_ratio": None, "debt_to_equity": None,
        "eps_growth_qoq": None, "revenue_growth_yoy": None,
        "dividend_yield": None, "short_float": None,
        "analyst_rating": "N/A", "num_analyst_opinions": None,
        "target_mean_price": None, "target_high_price": None,
        "target_low_price": None,
        "52w_high": None, "52w_low": None, "beta": None,
        "free_cashflow": None, "total_revenue": None,
        "next_earnings_date": None,
    }

    # ── Tier 0: raw OHLC download — always works ─────────────────────────────
    try:
        df1y = _download(symbol, period="1y")
        if not df1y.empty:
            if "Close" in df1y.columns:
                result["current_price"] = _safe_float(df1y["Close"].iloc[-1])
            if "High" in df1y.columns:
                result["52w_high"] = _safe_float(df1y["High"].max())
            if "Low" in df1y.columns:
                result["52w_low"] = _safe_float(df1y["Low"].min())
            # Beta: computed from 1y daily returns vs SPY — no extra API needed
            try:
                spy1y = _download("SPY", period="1y")
                if not spy1y.empty:
                    s_ret = df1y["Close"].astype(float).pct_change().dropna()
                    m_ret = spy1y["Close"].astype(float).pct_change().dropna()
                    both  = pd.concat([s_ret, m_ret], axis=1).dropna()
                    if len(both) > 20:
                        cov_val = float(both.iloc[:, 0].cov(both.iloc[:, 1]))
                        var_val = float(both.iloc[:, 1].var())
                        if var_val > 0:
                            result["beta"] = round(cov_val / var_val, 2)
            except Exception:
                pass
    except Exception:
        pass

    # ── Tier 1: fast_info (needs curl_cffi; skip silently if unavailable) ────
    try:
        fi = yf.Ticker(symbol).fast_info
        result["market_cap"] = _safe_float(getattr(fi, "market_cap", None))
        result["currency"]   = getattr(fi, "currency", "USD") or "USD"
        if result["current_price"] is None:
            result["current_price"] = _safe_float(getattr(fi, "last_price", None))
        # yfinance 1.x renamed these; try both attribute names
        if result["52w_high"] is None:
            result["52w_high"] = _safe_float(
                getattr(fi, "year_high", None) or getattr(fi, "fifty_two_week_high", None))
        if result["52w_low"] is None:
            result["52w_low"] = _safe_float(
                getattr(fi, "year_low", None) or getattr(fi, "fifty_two_week_low", None))
    except Exception as e:
        print(f"[TickerIQ] fast_info {symbol}: {type(e).__name__}: {e}")

    # ── Tier 2: .info — rich fundamentals (needs curl_cffi on cloud) ─────────
    try:
        info = yf.Ticker(symbol).info
        if info and len(info) > 5:
            def sf(key, dec=2):
                return _safe_float(info.get(key), dec)
            result["name"]                 = info.get("longName", symbol)
            result["sector"]               = info.get("sector", "Unknown")
            result["industry"]             = info.get("industry", "Unknown")
            result["market_cap"]           = result["market_cap"] or info.get("marketCap")
            result["currency"]             = info.get("currency", result["currency"])
            result["current_price"]        = result["current_price"] or sf("currentPrice") or sf("regularMarketPrice")
            result["pe_trailing"]          = sf("trailingPE")
            result["pe_forward"]           = sf("forwardPE")
            result["pb_ratio"]             = sf("priceToBook")
            result["peg_ratio"]            = sf("pegRatio")
            result["debt_to_equity"]       = sf("debtToEquity")
            result["eps_growth_qoq"]       = sf("earningsQuarterlyGrowth", 4)
            result["revenue_growth_yoy"]   = sf("revenueGrowth", 4)
            # yfinance inconsistently returns dividendYield as a ratio (0.067)
            # or as a percentage (6.7) depending on version/ticker — normalise to ratio.
            dy = sf("dividendYield", 4)
            if dy is not None and dy > 1.0:
                dy = round(dy / 100, 4)
            result["dividend_yield"]       = dy
            result["short_float"]          = sf("shortPercentOfFloat", 4)
            result["analyst_rating"]       = info.get("recommendationKey", "N/A")
            result["num_analyst_opinions"] = info.get("numberOfAnalystOpinions")
            result["target_mean_price"]    = sf("targetMeanPrice")
            result["target_high_price"]    = sf("targetHighPrice")
            result["target_low_price"]     = sf("targetLowPrice")
            result["52w_high"]             = result["52w_high"] or sf("fiftyTwoWeekHigh")
            result["52w_low"]              = result["52w_low"]  or sf("fiftyTwoWeekLow")
            result["beta"]                 = sf("beta") or result["beta"]
            result["free_cashflow"]        = info.get("freeCashflow")
            result["total_revenue"]        = info.get("totalRevenue")
            # Earnings date — prefer the window-start timestamp; fall back to
            # earningsTimestamp.  Value is a Unix epoch int (seconds).
            try:
                et = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
                if isinstance(et, list):
                    et = et[0] if et else None
                if et and isinstance(et, (int, float)) and et > 0:
                    result["next_earnings_date"] = (
                        datetime.datetime.utcfromtimestamp(et).date()
                    )
            except Exception:
                pass
    except Exception as e:
        print(f"[TickerIQ] .info {symbol}: {type(e).__name__}: {e}")

    # ── Tier 3: yf.Search — lightweight fallback for sector/name on cloud ────
    if result["sector"] == "Unknown":
        try:
            search = yf.Search(symbol, max_results=1)
            quotes = getattr(search, "quotes", None) or []
            if quotes:
                q = quotes[0]
                if result["name"] == symbol:
                    result["name"] = q.get("longname") or q.get("shortname") or symbol
                result["sector"]   = q.get("sector")   or "Unknown"
                result["industry"] = q.get("industry") or "Unknown"
        except Exception as e:
            print(f"[TickerIQ] Search {symbol}: {type(e).__name__}: {e}")

    return result


# ── Master function ───────────────────────────────────────────────────────────

def get_all_metrics(symbol: str, trading_style: str) -> dict:
    """
    Fetches and returns all metrics needed for scoring and display.
    This is the single entry point called from app.py.
    """
    symbol = symbol.strip().upper()
    period = STYLE_PERIOD.get(trading_style, "6mo")

    fundamentals = get_fundamentals(symbol)
    sector       = fundamentals.get("sector", "Unknown")
    sector_etf   = SECTOR_ETF_MAP.get(sector)

    technicals   = get_technical_indicators(symbol, period=period)
    options      = get_options_sentiment(symbol)

    stock_return  = get_stock_return(symbol, period=period)
    spy_return    = get_stock_return("SPY",    period=period)
    sector_return = get_stock_return(sector_etf, period=period) if sector_etf else None

    return {
        "symbol":        symbol,
        "trading_style": trading_style,
        "sector":        sector,
        "sector_etf":    sector_etf,
        "technicals":    technicals,
        "fundamentals":  fundamentals,
        "options":       options,
        "performance": {
            "stock_return":  stock_return,
            "spy_return":    spy_return,
            "sector_return": sector_return,
            "period":        period,
        },
    }
