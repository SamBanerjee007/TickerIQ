"""
TickerIQ — Support & Resistance module.

Two methods:
  1. Pivot Points  — standard formula from prior day's OHLC (same methodology as eTrade)
                     PP = (H+L+C)/3  |  R1/R2/R3  |  S1/S2/S3
  2. Rolling S/R   — 52-week H/L, 20-day H/L, and recent swing highs/lows
"""

import numpy as np
import yfinance as yf


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download(symbol: str, period: str, interval: str = "1d"):
    import pandas as pd
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _r(v, dec=2):
    try:
        return round(float(v), dec)
    except Exception:
        return None


# ── Pivot Points ──────────────────────────────────────────────────────────────

def get_pivot_points(symbol: str) -> dict:
    """
    Standard Pivot Points calculated from the prior completed trading day.
    Formula used by eTrade and most professional platforms:
        PP = (H + L + C) / 3
        R1 = 2*PP - L     S1 = 2*PP - H
        R2 = PP + (H - L) S2 = PP - (H - L)
        R3 = H + 2*(PP-L) S3 = L - 2*(H-PP)
    """
    try:
        df = _download(symbol, period="5d")
        if df.empty or len(df) < 2:
            return {}

        prev = df.iloc[-2]
        H = float(prev["High"])
        L = float(prev["Low"])
        C = float(prev["Close"])
        HL = H - L

        PP = _r((H + L + C) / 3)
        R1 = _r(2 * PP - L)
        R2 = _r(PP + HL)
        R3 = _r(H + 2 * (PP - L))
        S1 = _r(2 * PP - H)
        S2 = _r(PP - HL)
        S3 = _r(L - 2 * (H - PP))

        return {"PP": PP, "R1": R1, "R2": R2, "R3": R3,
                "S1": S1, "S2": S2, "S3": S3}
    except Exception:
        return {}


# ── Rolling S/R ───────────────────────────────────────────────────────────────

def get_rolling_sr(symbol: str) -> dict:
    """
    Rolling window support & resistance:
      - 52-week high/low
      - 20-day high/low
      - Recent swing highs/lows (local extremes in the last 60 trading days)
    """
    try:
        df = _download(symbol, period="1y")
        if df.empty or len(df) < 30:
            return {}

        close = df["Close"].values.astype(float)
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)

        w52_high = _r(np.max(high))
        w52_low  = _r(np.min(low))
        d20_high = _r(np.max(high[-20:]))
        d20_low  = _r(np.min(low[-20:]))
        current  = _r(close[-1])

        # Swing highs/lows via simple peak detection (lookback = 2 bars each side)
        recent_h = high[-60:] if len(high) >= 60 else high
        recent_l = low[-60:]  if len(low)  >= 60 else low
        n = len(recent_h)

        swing_highs, swing_lows = [], []
        for i in range(2, n - 2):
            if (recent_h[i] > recent_h[i-1] and recent_h[i] > recent_h[i-2]
                    and recent_h[i] > recent_h[i+1] and recent_h[i] > recent_h[i+2]):
                swing_highs.append(_r(recent_h[i]))
            if (recent_l[i] < recent_l[i-1] and recent_l[i] < recent_l[i-2]
                    and recent_l[i] < recent_l[i+1] and recent_l[i] < recent_l[i+2]):
                swing_lows.append(_r(recent_l[i]))

        # Deduplicate and keep 3 closest to current price
        swing_highs = sorted(set(swing_highs), key=lambda x: abs(x - current))[:3]
        swing_lows  = sorted(set(swing_lows),  key=lambda x: abs(x - current))[:3]

        return {
            "current_price": current,
            "52w_high":      w52_high,
            "52w_low":       w52_low,
            "20d_high":      d20_high,
            "20d_low":       d20_low,
            "swing_highs":   sorted(swing_highs, reverse=True),
            "swing_lows":    sorted(swing_lows,  reverse=True),
        }
    except Exception:
        return {}


# ── Combined entry point ──────────────────────────────────────────────────────

def get_sr_context(symbol: str) -> dict:
    """Returns both pivot points and rolling S/R merged into one dict."""
    return {
        "pivots":  get_pivot_points(symbol),
        "rolling": get_rolling_sr(symbol),
    }


# ── Scoring helper ────────────────────────────────────────────────────────────

def score_price_vs_sr(current_price: float | None, pivots: dict) -> int:
    """
    Returns a 1-5 score based on where current price sits relative to pivot levels.

    5 — Strongly above PP and R1, momentum is bullish
    4 — Between PP and R1 (bullish zone)
    3 — Near PP (neutral pivot)
    2 — Between S1 and PP (mildly bearish)
    1 — Below S1 or S2 (bearish breakdown territory)
    """
    if not pivots or current_price is None:
        return 3

    pp = pivots.get("PP", current_price)
    r1 = pivots.get("R1", current_price * 1.02)
    r2 = pivots.get("R2", current_price * 1.04)
    s1 = pivots.get("S1", current_price * 0.98)
    s2 = pivots.get("S2", current_price * 0.96)

    if current_price >= r2:
        return 3   # Extended above R2 — overbought, resistance overhead
    if current_price >= r1:
        return 4   # Between R1 and R2 — strong momentum
    if current_price >= pp:
        return 4   # Above PP — bullish structure
    if current_price >= s1:
        return 2   # Below PP, above S1 — weakening
    if current_price >= s2:
        return 2   # Below S1 — bearish
    return 1       # Below S2 — breakdown
