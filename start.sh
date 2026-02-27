#!/bin/bash
# Writes .streamlit/secrets.toml from Railway environment variables,
# then starts the Streamlit app.
set -e

mkdir -p .streamlit

cat > .streamlit/secrets.toml << SECRETS
[auth]
username = "${AUTH_USERNAME}"
password = "${AUTH_PASSWORD}"

[supabase]
url      = "${SUPABASE_URL}"
anon_key = "${SUPABASE_ANON_KEY}"
SECRETS

exec streamlit run app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
