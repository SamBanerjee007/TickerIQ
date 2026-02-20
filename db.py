"""
TickerIQ — Persistence layer.

Primary store: Supabase (when configured in secrets.toml).
Fallback store: local SQLite file (tickeriq_local.db) — works out of the box
                with zero configuration, perfect for local development.

Handles:
  - Logging every analysis query (symbol, style, score, signal, ip_hash)
  - IP-based rate limiting (max N queries per hour from one IP)
  - Trending tickers (most queried in the last 24 h)
  - Admin usage statistics
"""

import datetime
import hashlib
import pathlib
import sqlite3
import threading

import pandas as pd
import streamlit as st


# ── Local SQLite fallback ─────────────────────────────────────────────────────

_DB_PATH = pathlib.Path(__file__).parent / "tickeriq_local.db"
_local_lock = threading.Lock()


def _local_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            symbol       TEXT    NOT NULL,
            trading_style TEXT,
            score        REAL,
            signal       TEXT,
            ip_hash      TEXT
        )
    """)
    conn.commit()
    return conn


# ── Supabase client (cached, optional) ───────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_supabase():
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["anon_key"]
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


# ── IP helpers ────────────────────────────────────────────────────────────────

_SALT = "tickeriq-v1"


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(f"{ip}{_SALT}".encode()).hexdigest()[:20]


def _client_ip() -> str:
    return st.session_state.get("_client_ip", "unknown")


# ── Public API ────────────────────────────────────────────────────────────────

def log_query(symbol: str, trading_style: str, score: float, signal: str) -> None:
    """Log one analysis. Uses Supabase if configured, otherwise local SQLite."""
    ip_hash = _hash_ip(_client_ip())
    sb = _get_supabase()
    if sb is not None:
        try:
            sb.table("queries").insert({
                "symbol":        symbol.upper(),
                "trading_style": trading_style,
                "score":         score,
                "signal":        signal,
                "ip_hash":       ip_hash,
            }).execute()
        except Exception:
            pass
        return

    # SQLite fallback
    with _local_lock:
        try:
            conn = _local_conn()
            conn.execute(
                "INSERT INTO queries (symbol, trading_style, score, signal, ip_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol.upper(), trading_style, score, signal, ip_hash),
            )
            conn.commit()
        except Exception:
            pass


def check_rate_limit(max_per_hour: int = 20) -> bool:
    """Returns True if the request is within limit. Fails open on error."""
    ip_hash = _hash_ip(_client_ip())
    cutoff  = (datetime.datetime.utcnow() - datetime.timedelta(hours=1)).isoformat()

    sb = _get_supabase()
    if sb is not None:
        try:
            result = (
                sb.table("queries")
                  .select("id", count="exact")
                  .eq("ip_hash", ip_hash)
                  .gte("created_at", cutoff)
                  .execute()
            )
            return (result.count or 0) < max_per_hour
        except Exception:
            return True

    # SQLite fallback
    with _local_lock:
        try:
            conn = _local_conn()
            row = conn.execute(
                "SELECT COUNT(*) FROM queries WHERE ip_hash=? AND created_at>=?",
                (ip_hash, cutoff),
            ).fetchone()
            return (row[0] if row else 0) < max_per_hour
        except Exception:
            return True


def get_trending(hours: int = 24, limit: int = 5) -> list[dict]:
    """Most-queried tickers in the last `hours` hours."""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=hours)).isoformat()

    sb = _get_supabase()
    if sb is not None:
        try:
            result = (
                sb.table("queries")
                  .select("symbol, signal")
                  .gte("created_at", cutoff)
                  .execute()
            )
            if not result.data:
                return []
            return _aggregate_trending(pd.DataFrame(result.data), limit)
        except Exception:
            return []

    # SQLite fallback
    with _local_lock:
        try:
            conn = _local_conn()
            df = pd.read_sql_query(
                "SELECT symbol, signal FROM queries WHERE created_at >= ?",
                conn, params=(cutoff,),
            )
            if df.empty:
                return []
            return _aggregate_trending(df, limit)
        except Exception:
            return []


def _aggregate_trending(df: pd.DataFrame, limit: int) -> list[dict]:
    trending = (
        df.groupby("symbol")
          .agg(
              count=("symbol", "count"),
              top_signal=("signal", lambda x: x.mode().iloc[0] if len(x) > 0 else "Hold"),
          )
          .reset_index()
          .sort_values("count", ascending=False)
          .head(limit)
    )
    return trending.to_dict("records")


def get_admin_stats() -> dict:
    """Returns stats dict for the admin dashboard. Empty dict on failure."""
    sb = _get_supabase()
    if sb is not None:
        try:
            result = sb.table("queries").select("*").execute()
            if not result.data:
                return {}
            return _build_stats(pd.DataFrame(result.data))
        except Exception:
            return {}

    # SQLite fallback
    with _local_lock:
        try:
            conn = _local_conn()
            df = pd.read_sql_query("SELECT * FROM queries", conn)
            if df.empty:
                return {}
            return _build_stats(df)
        except Exception:
            return {}


def _build_stats(df: pd.DataFrame) -> dict:
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    now = pd.Timestamp.now(tz="UTC")

    today_mask  = df["created_at"] >= pd.Timestamp(now.date()).tz_localize("UTC")
    week_mask   = df["created_at"] >= (now - pd.Timedelta(days=7))
    month_mask  = df["created_at"] >= (now - pd.Timedelta(days=30))

    top_sym = (
        df["symbol"].value_counts()
          .head(20)
          .reset_index()
          .rename(columns={"index": "symbol", "symbol": "queries", "count": "queries"})
    )
    # Normalise column names across pandas versions
    top_sym.columns = ["symbol", "queries"]

    by_style = df["trading_style"].value_counts().reset_index()
    by_style.columns = ["trading_style", "count"]

    by_signal = df["signal"].value_counts().reset_index()
    by_signal.columns = ["signal", "count"]

    daily = (
        df.set_index("created_at")
          .resample("D")
          .size()
          .reset_index(name="queries")
          .rename(columns={"created_at": "date"})
    )

    return {
        "raw":          df,
        "total":        len(df),
        "today":        int(today_mask.sum()),
        "this_week":    int(week_mask.sum()),
        "this_month":   int(month_mask.sum()),
        "top_symbols":  top_sym,
        "by_style":     by_style,
        "by_signal":    by_signal,
        "daily_volume": daily,
    }
