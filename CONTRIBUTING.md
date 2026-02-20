# Contributing to TickerIQ

Thank you for considering a contribution! Here are the guidelines.

## How to contribute

1. **Open an issue first** for any non-trivial change (new metric, scoring tweak, UI improvement).
2. Fork the repo and create a descriptive branch: `feature/bollinger-divergence`, `fix/rsi-edge-case`.
3. Keep changes focused — one feature or fix per PR.
4. Test your changes locally (`streamlit run app.py`) before opening a PR.
5. Open a Pull Request against `main` with a clear description of what changed and why.

## Good first issues

- Adding a new sector ETF to `SECTOR_ETF_MAP`
- Improving scoring thresholds with cited research
- Adding a new chart type to the Admin dashboard
- UI/UX improvements (mobile layout, color palette, tooltips)

## Code style

- Python: follow PEP 8, use type hints where practical
- Keep functions small and single-purpose
- No secrets or API keys in code — use `st.secrets`

## Data sources

TickerIQ intentionally uses only free, public data via `yfinance`.
PRs that add paid API dependencies will not be accepted without discussion.
