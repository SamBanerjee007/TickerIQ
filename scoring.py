"""
TickerIQ — Weighted scoring engine.

Each metric is scored independently on a 1–5 scale:
  1 = Very bearish / expensive / weak
  3 = Neutral
  5 = Very bullish / cheap / strong

Component scores are then combined using trading-style-specific weights
to produce a final composite score (1–5) and a Buy/Hold/Sell signal.
"""

import math

from support_resistance import score_price_vs_sr

# ── Per-style weights (must sum to 1.0) ───────────────────────────────────────

STYLE_WEIGHTS: dict[str, dict[str, float]] = {
    "Swing Trader": {
        "rsi":             0.12,
        "macd":            0.12,
        "bollinger":       0.08,
        "volume":          0.06,
        "adx":             0.04,
        "sr_level":        0.10,
        "options_pcr":     0.15,
        "short_interest":  0.05,
        "analyst":         0.10,
        "vs_spy":          0.10,
        "vs_sector":       0.08,
    },
    "Position Trader": {
        "rsi":             0.07,
        "macd":            0.07,
        "bollinger":       0.04,
        "volume":          0.03,
        "adx":             0.03,
        "sr_level":        0.05,
        "options_pcr":     0.08,
        "short_interest":  0.05,
        "pe_ratio":        0.10,
        "eps_growth":      0.13,
        "revenue_growth":  0.07,
        "analyst":         0.12,
        "vs_spy":          0.08,
        "vs_sector":       0.08,
    },
    "Long-term Investor": {
        "rsi":             0.02,
        "macd":            0.02,
        "bollinger":       0.01,
        "volume":          0.01,
        "pe_ratio":        0.12,
        "forward_pe":      0.08,
        "pb_ratio":        0.06,
        "peg_ratio":       0.08,
        "eps_growth":      0.15,
        "revenue_growth":  0.10,
        "dividend_yield":  0.07,
        "debt_equity":     0.06,
        "analyst":         0.10,
        "vs_spy":          0.12,
    },
}

# ── Individual metric scorers (each returns int 1–5) ─────────────────────────

def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def score_rsi(rsi) -> int:
    if _nan(rsi):
        return 3
    if rsi < 25:   return 2   # extreme oversold — downtrend risk
    if rsi < 38:   return 4   # oversold recovery zone — buy signal
    if rsi < 52:   return 4   # bullish momentum zone
    if rsi < 65:   return 3   # neutral / mid
    if rsi < 75:   return 2   # overbought
    return 1                   # extreme overbought


def score_macd(macd, signal) -> int:
    if _nan(macd) or _nan(signal):
        return 3
    diff = macd - signal
    if diff >  0.5:  return 5
    if diff >  0.1:  return 4
    if diff >= 0.0:  return 3
    if diff > -0.1:  return 2
    return 1


def score_bollinger(bb_pct) -> int:
    # bb_pct: 0 = at lower band (oversold), 100 = at upper band (overbought)
    if _nan(bb_pct):
        return 3
    if bb_pct < 10:  return 5   # near/below lower band
    if bb_pct < 30:  return 4   # below midline
    if bb_pct < 65:  return 3   # middle zone
    if bb_pct < 85:  return 2   # approaching upper band
    return 1                     # at/above upper band


def score_volume(vol_ratio) -> int:
    if _nan(vol_ratio) or vol_ratio is None:
        return 3
    if vol_ratio >= 2.5:  return 4   # significant volume spike
    if vol_ratio >= 1.4:  return 4   # above-average — confirmed move
    if vol_ratio >= 0.7:  return 3   # normal range
    return 2                          # low volume — weak conviction


def score_adx(adx) -> int:
    if _nan(adx):
        return 3
    if adx >= 40:  return 5   # very strong trend
    if adx >= 25:  return 4   # trending market
    if adx >= 20:  return 3   # developing trend
    return 2                   # weak / choppy


def score_pcr(pc_ratio) -> int:
    if _nan(pc_ratio) or pc_ratio is None:
        return 3
    if pc_ratio < 0.5:   return 5
    if pc_ratio < 0.7:   return 4
    if pc_ratio < 1.0:   return 3
    if pc_ratio < 1.5:   return 2
    return 1


def score_short_interest(short_float) -> int:
    # short_float is decimal (0.05 = 5%)
    if _nan(short_float) or short_float is None:
        return 3
    pct = short_float * 100
    if pct < 3:    return 4   # very low short interest — low bearish conviction
    if pct < 8:    return 3   # normal
    if pct < 15:   return 2   # elevated — significant bearish pressure
    return 1                   # very high — strong bearish or squeeze risk


def score_pe(pe) -> int:
    if _nan(pe) or pe is None:
        return 3
    if pe <= 0:    return 1   # negative earnings
    if pe < 10:    return 5
    if pe < 18:    return 4
    if pe < 28:    return 3
    if pe < 45:    return 2
    return 1


def score_forward_pe(fpe) -> int:
    if _nan(fpe) or fpe is None:
        return 3
    if fpe <= 0:   return 1
    if fpe < 12:   return 5
    if fpe < 20:   return 4
    if fpe < 28:   return 3
    if fpe < 40:   return 2
    return 1


def score_pb(pb) -> int:
    if _nan(pb) or pb is None:
        return 3
    if pb <= 0:    return 1
    if pb < 1.0:   return 5
    if pb < 2.0:   return 4
    if pb < 4.0:   return 3
    if pb < 8.0:   return 2
    return 1


def score_peg(peg) -> int:
    if _nan(peg) or peg is None:
        return 3
    if peg <= 0:   return 1
    if peg < 0.5:  return 5
    if peg < 1.0:  return 4
    if peg < 1.5:  return 3
    if peg < 2.5:  return 2
    return 1


def score_eps_growth(eps_growth) -> int:
    # decimal: 0.15 = 15% quarterly growth
    if _nan(eps_growth) or eps_growth is None:
        return 3
    pct = eps_growth * 100
    if pct >= 30:   return 5
    if pct >= 15:   return 4
    if pct >= 5:    return 3
    if pct >= 0:    return 2
    return 1


def score_revenue_growth(rev_growth) -> int:
    if _nan(rev_growth) or rev_growth is None:
        return 3
    pct = rev_growth * 100
    if pct >= 25:   return 5
    if pct >= 12:   return 4
    if pct >= 5:    return 3
    if pct >= 0:    return 2
    return 1


def score_dividend_yield(dy) -> int:
    if _nan(dy) or dy is None or dy == 0:
        return 3   # no dividend = neutral (growth focus)
    pct = dy * 100
    if pct >= 5:   return 5
    if pct >= 3:   return 4
    if pct >= 1.5: return 3
    return 2


def score_debt_equity(de) -> int:
    # yfinance returns D/E as a percentage (e.g. 150 = 1.5×)
    if _nan(de) or de is None:
        return 3
    ratio = de / 100
    if ratio < 0.2:   return 5
    if ratio < 0.5:   return 4
    if ratio < 1.0:   return 3
    if ratio < 2.0:   return 2
    return 1


def score_analyst(rating, target_price, current_price) -> int:
    mapping = {
        "strongbuy":    5,
        "buy":          4,
        "hold":         3,
        "underperform": 2,
        "sell":         1,
    }
    base = mapping.get(str(rating).lower().replace(" ", ""), 3) if rating else 3

    if target_price and current_price and current_price > 0:
        upside = (target_price - current_price) / current_price * 100
        if upside > 30:    base = min(5, base + 1)
        elif upside > 15:  base = min(5, base + 0)
        elif upside > 5:   pass
        elif upside > -5:  base = max(1, base - 0)
        else:              base = max(1, base - 1)

    return int(base)


def score_vs_benchmark(stock_ret, bench_ret) -> int:
    if _nan(stock_ret) or _nan(bench_ret) or stock_ret is None or bench_ret is None:
        return 3
    diff = stock_ret - bench_ret
    if diff > 15:    return 5
    if diff > 5:     return 4
    if diff > -2:    return 3
    if diff > -10:   return 2
    return 1


# ── Master scoring function ───────────────────────────────────────────────────

def calculate_score(metrics: dict, sr_data: dict) -> dict:
    """
    Accepts the output of get_all_metrics() and get_sr_context().
    Returns:
        total_score       float (1–5)
        signal            str  e.g. "Buy"
        signal_color      str  hex color
        component_scores  dict {metric: {score, weight, label}}
    """
    trading_style = metrics.get("trading_style", "Position Trader")
    weights       = STYLE_WEIGHTS.get(trading_style, STYLE_WEIGHTS["Position Trader"])

    tech  = metrics.get("technicals",  {})
    fund  = metrics.get("fundamentals", {})
    opts  = metrics.get("options",      {})
    perf  = metrics.get("performance",  {})
    pivots = sr_data.get("pivots", {})

    current_price = tech.get("current_price") or fund.get("current_price")

    # ── Compute all component scores ──────────────────────────────────────────
    raw: dict[str, int] = {
        "rsi":            score_rsi(tech.get("rsi")),
        "macd":           score_macd(tech.get("macd"), tech.get("macd_signal")),
        "bollinger":      score_bollinger(tech.get("bb_pct")),
        "volume":         score_volume(tech.get("volume_ratio")),
        "adx":            score_adx(tech.get("adx")),
        "sr_level":       score_price_vs_sr(current_price, pivots),
        "options_pcr":    score_pcr(opts.get("pc_ratio")),
        "short_interest": score_short_interest(fund.get("short_float")),
        "pe_ratio":       score_pe(fund.get("pe_trailing")),
        "forward_pe":     score_forward_pe(fund.get("pe_forward")),
        "pb_ratio":       score_pb(fund.get("pb_ratio")),
        "peg_ratio":      score_peg(fund.get("peg_ratio")),
        "eps_growth":     score_eps_growth(fund.get("eps_growth_qoq")),
        "revenue_growth": score_revenue_growth(fund.get("revenue_growth_yoy")),
        "dividend_yield": score_dividend_yield(fund.get("dividend_yield")),
        "debt_equity":    score_debt_equity(fund.get("debt_to_equity")),
        "analyst":        score_analyst(
                              fund.get("analyst_rating"),
                              fund.get("target_mean_price"),
                              current_price
                          ),
        "vs_spy":         score_vs_benchmark(perf.get("stock_return"), perf.get("spy_return")),
        "vs_sector":      score_vs_benchmark(perf.get("stock_return"), perf.get("sector_return")),
    }

    # ── Weighted average ──────────────────────────────────────────────────────
    total = weight_sum = 0.0
    component_scores = {}

    for key, weight in weights.items():
        score = raw.get(key, 3)
        total       += score * weight
        weight_sum  += weight
        component_scores[key] = {
            "score":   score,
            "weight":  weight,
            "label":   _METRIC_LABELS.get(key, key.replace("_", " ").title()),
        }

    final = round(total / weight_sum if weight_sum > 0 else 3.0, 2)

    return {
        "total_score":      final,
        "signal":           _signal(final),
        "signal_color":     _signal_color(final),
        "component_scores": component_scores,
        "raw_scores":       raw,
    }


# ── Signal thresholds & labels ────────────────────────────────────────────────

def _signal(score: float) -> str:
    if score >= 4.0:   return "Strong Buy"
    if score >= 3.25:  return "Buy"
    if score >= 2.75:  return "Hold"
    if score >= 2.0:   return "Sell"
    return "Strong Sell"


def _signal_color(score: float) -> str:
    if score >= 4.0:   return "#00C853"
    if score >= 3.25:  return "#69F0AE"
    if score >= 2.75:  return "#FFD600"
    if score >= 2.0:   return "#FF6D00"
    return "#D50000"


_METRIC_LABELS = {
    "rsi":            "RSI (14-day)",
    "macd":           "MACD Signal",
    "bollinger":      "Bollinger Bands",
    "volume":         "Volume Trend",
    "adx":            "ADX Trend Strength",
    "sr_level":       "Support / Resistance",
    "options_pcr":    "Put/Call Ratio",
    "short_interest": "Short Interest",
    "pe_ratio":       "P/E Ratio (TTM)",
    "forward_pe":     "Forward P/E",
    "pb_ratio":       "P/B Ratio",
    "peg_ratio":      "PEG Ratio",
    "eps_growth":     "EPS Growth (QoQ)",
    "revenue_growth": "Revenue Growth (YoY)",
    "dividend_yield": "Dividend Yield",
    "debt_equity":    "Debt / Equity",
    "analyst":        "Analyst Consensus",
    "vs_spy":         "vs S&P 500 (SPY)",
    "vs_sector":      "vs Sector ETF",
}
