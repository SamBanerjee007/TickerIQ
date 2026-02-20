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
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*utcnow.*")   # yfinance uses deprecated Timestamp.utcnow

# ── Shared HTTP session ────────────────────────────────────────────────────────
# Cloud platforms (Streamlit, AWS) are often rate-limited by Yahoo Finance.
# Passing a session with a real browser User-Agent bypasses this reliably.

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.5",
})

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
        stock = yf.Ticker(symbol, session=_SESSION)
        if not stock.options:
            return {"pc_ratio": None, "sentiment": "N/A", "calls_oi": 0, "puts_oi": 0}

        today   = datetime.date.today()
        cutoff  = today + datetime.timedelta(days=90)
        calls_oi = puts_oi = 0

        dates = [d for d in stock.options
                 if today <= datetime.datetime.strptime(d, "%Y-%m-%d").date() <= cutoff]

        for d in dates:
            try:
                chain     = stock.option_chain(d)
                calls_oi += int(chain.calls["openInterest"].sum())
                puts_oi  += int(chain.puts["openInterest"].sum())
            except Exception:
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
    except Exception:
        return {"pc_ratio": None, "sentiment": "N/A", "calls_oi": 0, "puts_oi": 0}


# ── Fundamentals ──────────────────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    try:
        info = yf.Ticker(symbol, session=_SESSION).info

        def sf(key, dec=2):
            return _safe_float(info.get(key), dec)

        return {
            "symbol":              symbol,
            "name":                info.get("longName", symbol),
            "sector":              info.get("sector", "Unknown"),
            "industry":            info.get("industry", "Unknown"),
            "market_cap":          info.get("marketCap"),
            "currency":            info.get("currency", "USD"),
            "current_price":       sf("currentPrice") or sf("regularMarketPrice"),
            "pe_trailing":         sf("trailingPE"),
            "pe_forward":          sf("forwardPE"),
            "pb_ratio":            sf("priceToBook"),
            "peg_ratio":           sf("pegRatio"),
            "debt_to_equity":      sf("debtToEquity"),
            "eps_growth_qoq":      sf("earningsQuarterlyGrowth", 4),
            "revenue_growth_yoy":  sf("revenueGrowth", 4),
            "dividend_yield":      sf("dividendYield", 4),
            "short_float":         sf("shortPercentOfFloat", 4),
            "analyst_rating":      info.get("recommendationKey", "N/A"),
            "num_analyst_opinions": info.get("numberOfAnalystOpinions"),
            "target_mean_price":   sf("targetMeanPrice"),
            "target_high_price":   sf("targetHighPrice"),
            "target_low_price":    sf("targetLowPrice"),
            "52w_high":            sf("fiftyTwoWeekHigh"),
            "52w_low":             sf("fiftyTwoWeekLow"),
            "beta":                sf("beta"),
            "free_cashflow":       info.get("freeCashflow"),
            "total_revenue":       info.get("totalRevenue"),
        }
    except Exception:
        return {"sector": "Unknown", "name": symbol}


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
