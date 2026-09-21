# Dynamic NSE AI Paper-Trading Bot

This repository is a **paper-only** Upstox + Gemini intraday research bot. It deliberately contains **no order-placement implementation**. A real Upstox production access token may be used for OAuth and market-data APIs, but entries and exits are simulated locally.

## What the final version does

1. Starts Telegram polling.
2. Performs the Upstox OAuth authorization-code flow and saves the daily access token locally.
3. Validates V3 market-data authorization before opening the WebSocket.
4. Downloads the current Upstox NSE instrument master.
5. Uses the dynamic NSE equity universe; no hardcoded stock list is used.
6. Warms each stock with cached/historical 1-minute candles before live scanning.
7. Maintains a SQLite market-history cache across GitHub Actions runs.
8. Uses prior candles for EMA/RSI/volume baselines immediately; current-session bars build today's VWAP and 5-minute momentum.
9. Subscribes to live LTPC data for the dynamic stock universe plus NIFTY 50 when it can be resolved from the instrument master.
10. Ranks qualifying stocks in Python and sends only the top 15 to Gemini.
11. Gemini can return only a stock present in those 15 candidates.
12. Opens paper positions only, persists them to `data/open_positions.json`, updates live P&L, checks paper SL/target, and sends entry/exit/open-position notifications to Telegram.
13. Squares off at the configured session end.

## Local / PyCharm

Copy `.env.example` to `.env` and fill in:

- `BOT_TOKEN`
- `CHAT_ID`
- `GEMINI_API_KEY`
- `UPSTOX_CLIENT_ID`
- `UPSTOX_CLIENT_SECRET`
- `UPSTOX_REDIRECT_URI`

Keep `UPSTOX_OAUTH_REQUIRED=true` for the daily Telegram login flow.

Run:

```bash
python -m pip install -r requirements.txt
python trading_engine.py
```

The bot sends the Upstox login URL to Telegram. Complete the login and send the **full redirect URL** back to the bot. The bot exchanges the one-time authorization code for an access token and saves it in `data/access_token.txt`.

## GitHub Actions

The repository includes `.github/workflows/paper_trading.yml`.

Create these **repository secrets**:

- `BOT_TOKEN`
- `CHAT_ID`
- `GEMINI_API_KEY`
- `UPSTOX_CLIENT_ID`
- `UPSTOX_CLIENT_SECRET`
- `UPSTOX_REDIRECT_URI`

Do **not** create an `UPSTOX_ACCESS_TOKEN` secret for the normal workflow. The workflow performs the daily OAuth flow through Telegram.

The workflow starts at 09:10 IST on weekdays (plus manual `workflow_dispatch`). The Python engine blocks entries until 09:15 and, for GitHub-hosted testing, stops the paper session at 15:05 because GitHub-hosted jobs have a 6-hour execution limit. The normal local/VPS settings remain 09:15–15:30.

The workflow caches only `data/market_history.db`. It never caches `data/access_token.txt` and does not upload the token as an artifact.

## Historical warm-up

The first run may take several minutes to bootstrap historical 1-minute candles for the dynamic universe because the Upstox API has standard API rate limits. Subsequent GitHub runs restore the SQLite cache and usually need far fewer historical requests.

The scanner does **not** treat old VWAP as today's VWAP. Historical candles seed EMA/RSI/volume history; today's VWAP and 5-minute momentum are calculated from the current trading session.

## Safety

There is no `OrderApi`, `OrderApiV3`, `place_order`, order endpoint, or live broker execution module in this repository. `paper_trading_only` is hard-coded `True` in the engine.

## Data files

- `data/market_history.db` — historical/live 1-minute cache
- `data/open_positions.json` — current paper positions
- `data/paper_trades.jsonl` — paper entry/exit log
- `data/ai_decisions.jsonl` — scanner candidates + Gemini decisions
- `data/session_state.json` — paper session summary
- `data/engine.log` — GitHub Actions console log copied to the artifact

## Important GitHub Actions limitation

A GitHub-hosted job can run for at most 6 hours. The included workflow therefore intentionally ends the paper session at 15:05 IST. For an uninterrupted 09:15–15:30 session, move the same repository to a self-hosted runner/VPS; no trading-logic changes are required.

## Current-session bootstrap

The scanner does not wait for the bot to create the first five/six 1-minute bars. Historical candles warm EMA/RSI/volume indicators before the market session. During live market hours, the first live tick for an active stock triggers a V3 Intraday Candle API request for that stock, which seeds the current day's 1-minute `session_bars`; the live WebSocket then continues updating those bars. This avoids treating yesterday's data as today's VWAP/momentum while avoiding a full 2,000+ request intraday bootstrap. Upstox documents the V3 intraday endpoint at `/historical-candle/intraday/:instrument_key/minutes/1`.

The session bootstrap is capped by `SESSION_BOOTSTRAP_MAX_REQUESTS` (default 500) and shares the historical API rate limiter. Stocks that do not generate live activity are not fetched individually because they are unlikely to be useful intraday candidates.
