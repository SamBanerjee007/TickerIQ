-- TickerIQ — Supabase database setup
-- Run this once in the Supabase SQL Editor (Dashboard → SQL Editor → New query)

-- Queries table: one row per stock analysis run
CREATE TABLE IF NOT EXISTS queries (
    id             BIGSERIAL PRIMARY KEY,
    symbol         TEXT         NOT NULL,
    trading_style  TEXT,
    score          FLOAT,
    signal         TEXT,
    ip_hash        TEXT,                           -- hashed, never the raw IP
    created_at     TIMESTAMPTZ  DEFAULT NOW()
);

-- Indexes for trending and rate-limit queries
CREATE INDEX IF NOT EXISTS queries_created_at_idx ON queries (created_at DESC);
CREATE INDEX IF NOT EXISTS queries_symbol_idx     ON queries (symbol);
CREATE INDEX IF NOT EXISTS queries_ip_hash_idx    ON queries (ip_hash, created_at DESC);

-- Row Level Security
ALTER TABLE queries ENABLE ROW LEVEL SECURITY;

-- Allow the Streamlit app (using the anon key) to INSERT and SELECT
CREATE POLICY "anon_insert" ON queries
    FOR INSERT TO anon
    WITH CHECK (true);

CREATE POLICY "anon_select" ON queries
    FOR SELECT TO anon
    USING (true);
