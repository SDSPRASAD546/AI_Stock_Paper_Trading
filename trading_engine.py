from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlparse
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from ai_selector import AITradeDecision, GeminiStockSelector
from market_scanner import DEFAULT_INSTRUMENT_URL, DynamicStockScanner
from telegram_bot import TelegramBot

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
HISTORICAL_URL = "https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/1/{to_date}/{from_date}"
INTRADAY_URL = "https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/minutes/1"


@dataclass
class Settings:
    # AI
    gemini_api_key: str
    gemini_model: str = "gemini-3.1-flash-lite"
    ai_min_confidence: float = 0.70
    ai_call_cooldown_seconds: int = 60

    # Dynamic scanner
    scanner_top_n: int = 15
    scanner_min_score: float = 45.0
    scanner_min_volume_ratio: float = 1.10
    scanner_min_momentum_5m: float = 0.10
    scanner_max_rsi: float = 72.0
    scanner_min_bars: int = 20
    scanner_min_session_bars: int = 5

    # Paper risk
    starting_capital: float = 100000.0
    risk_per_trade_pct: float = 0.005
    max_allocation_pct: float = 0.12
    stop_loss_pct: float = 0.008
    target_pct: float = 0.016
    max_trades: int = 5
    cooldown_minutes: int = 10
    allow_short: bool = False

    # Hard safety: no live order executor exists anywhere in this project.
    paper_trading_only: bool = True

    # Upstox OAuth + market data
    upstox_client_id: str = ""
    upstox_client_secret: str = ""
    upstox_redirect_uri: str = ""
    upstox_oauth_required: bool = True
    upstox_access_token: str = ""
    upstox_access_token_file: str = "data/access_token.txt"
    instrument_file: str = "data/NSE.json.gz"
    instrument_url: str = DEFAULT_INSTRUMENT_URL
    auto_download_instruments: bool = True
    live_feed_enabled: bool = True
    upstox_mode: str = "ltpc"
    max_subscribed_instruments: int = 5000
    reconnect_interval_seconds: int = 5
    reconnect_attempts: int = 50

    # Session
    entry_start: str = "09:15"
    entry_end: str = "15:10"
    square_off: str = "15:15"
    market_end: str = "15:30"
    run_stop_time: str = ""

    # Historical warm-up
    history_db: str = "data/market_history.db"
    history_bars_per_instrument: int = 120
    history_days_back: int = 7
    history_refresh_stale_days: int = 7
    history_rate_limit_per_minute: int = 480
    session_bootstrap_enabled: bool = True
    session_bootstrap_max_requests: int = 500
    session_bootstrap_concurrency: int = 10
    session_bootstrap_min_bars: int = 6

    # Runtime
    evaluate_every_seconds: int = 60
    position_snapshot_seconds: int = 60
    persist_interval_seconds: int = 5
    telegram_scan_messages: bool = False
    output_dir: str = "data"

    @staticmethod
    def _bool(value: str | None, default: bool) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _clock(value: str) -> dt_time:
        hour, minute = map(int, value.strip().split(":", 1))
        return dt_time(hour=hour, minute=minute, tzinfo=IST)

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
        token_file = Path(os.getenv("UPSTOX_ACCESS_TOKEN_FILE", "data/access_token.txt"))
        if not access_token and token_file.exists():
            access_token = token_file.read_text(encoding="utf-8").strip()

        oauth_required = cls._bool(os.getenv("UPSTOX_OAUTH_REQUIRED"), True)
        client_id = os.getenv("UPSTOX_CLIENT_ID", "").strip()
        client_secret = os.getenv("UPSTOX_CLIENT_SECRET", "").strip()
        redirect_uri = os.getenv("UPSTOX_REDIRECT_URI", "").strip()
        if oauth_required:
            missing = [name for name, value in (
                ("UPSTOX_CLIENT_ID", client_id),
                ("UPSTOX_CLIENT_SECRET", client_secret),
                ("UPSTOX_REDIRECT_URI", redirect_uri),
            ) if not value]
            if missing:
                raise ValueError("Missing Upstox OAuth settings: " + ", ".join(missing))

        return cls(
            gemini_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            ai_min_confidence=float(os.getenv("AI_MIN_CONFIDENCE", "0.70")),
            ai_call_cooldown_seconds=int(os.getenv("AI_CALL_COOLDOWN_SECONDS", "60")),
            scanner_top_n=int(os.getenv("SCANNER_TOP_N", "15")),
            scanner_min_score=float(os.getenv("SCANNER_MIN_SCORE", "45")),
            scanner_min_volume_ratio=float(os.getenv("SCANNER_MIN_VOLUME_RATIO", "1.10")),
            scanner_min_momentum_5m=float(os.getenv("SCANNER_MIN_MOMENTUM_5M", "0.10")),
            scanner_max_rsi=float(os.getenv("SCANNER_MAX_RSI", "72")),
            scanner_min_bars=int(os.getenv("SCANNER_MIN_BARS", "20")),
            scanner_min_session_bars=int(os.getenv("SCANNER_MIN_SESSION_BARS", "5")),
            starting_capital=float(os.getenv("STARTING_CAPITAL", "100000")),
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.005")),
            max_allocation_pct=float(os.getenv("MAX_ALLOCATION_PCT", "0.12")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.008")),
            target_pct=float(os.getenv("TARGET_PCT", "0.016")),
            max_trades=int(os.getenv("MAX_TRADES", "5")),
            cooldown_minutes=int(os.getenv("COOLDOWN_MINUTES", "10")),
            allow_short=cls._bool(os.getenv("ALLOW_SHORT"), False),
            upstox_client_id=client_id,
            upstox_client_secret=client_secret,
            upstox_redirect_uri=redirect_uri,
            upstox_oauth_required=oauth_required,
            upstox_access_token=access_token,
            upstox_access_token_file=os.getenv("UPSTOX_ACCESS_TOKEN_FILE", "data/access_token.txt"),
            instrument_file=os.getenv("INSTRUMENT_FILE", "data/NSE.json.gz"),
            instrument_url=os.getenv("INSTRUMENT_URL", DEFAULT_INSTRUMENT_URL),
            auto_download_instruments=cls._bool(os.getenv("AUTO_DOWNLOAD_INSTRUMENTS"), True),
            live_feed_enabled=cls._bool(os.getenv("LIVE_FEED_ENABLED"), True),
            upstox_mode=os.getenv("UPSTOX_MODE", "ltpc"),
            max_subscribed_instruments=int(os.getenv("MAX_SUBSCRIBED_INSTRUMENTS", "5000")),
            reconnect_interval_seconds=int(os.getenv("RECONNECT_INTERVAL_SECONDS", "5")),
            reconnect_attempts=int(os.getenv("RECONNECT_ATTEMPTS", "50")),
            entry_start=os.getenv("ENTRY_START", "09:15"),
            entry_end=os.getenv("ENTRY_END", "15:10"),
            square_off=os.getenv("SQUARE_OFF", "15:15"),
            market_end=os.getenv("MARKET_END", "15:30"),
            run_stop_time=os.getenv("RUN_STOP_TIME", "").strip(),
            history_db=os.getenv("HISTORY_DB", "data/market_history.db"),
            history_bars_per_instrument=int(os.getenv("HISTORY_BARS_PER_INSTRUMENT", "120")),
            history_days_back=int(os.getenv("HISTORY_DAYS_BACK", "7")),
            history_refresh_stale_days=int(os.getenv("HISTORY_REFRESH_STALE_DAYS", "7")),
            history_rate_limit_per_minute=int(os.getenv("HISTORY_RATE_LIMIT_PER_MINUTE", "480")),
            session_bootstrap_enabled=cls._bool(os.getenv("SESSION_BOOTSTRAP_ENABLED"), True),
            session_bootstrap_max_requests=int(os.getenv("SESSION_BOOTSTRAP_MAX_REQUESTS", "500")),
            session_bootstrap_concurrency=int(os.getenv("SESSION_BOOTSTRAP_CONCURRENCY", "10")),
            session_bootstrap_min_bars=int(os.getenv("SESSION_BOOTSTRAP_MIN_BARS", "6")),
            evaluate_every_seconds=int(os.getenv("EVALUATE_EVERY_SECONDS", "60")),
            position_snapshot_seconds=int(os.getenv("POSITION_SNAPSHOT_SECONDS", "60")),
            persist_interval_seconds=int(os.getenv("PERSIST_INTERVAL_SECONDS", "5")),
            telegram_scan_messages=cls._bool(os.getenv("TELEGRAM_SCAN_MESSAGES"), False),
            output_dir=os.getenv("OUTPUT_DIR", "data"),
        )

    def clock(self, name: str) -> dt_time:
        return self._clock(getattr(self, name))


@dataclass
class PaperPosition:
    symbol: str
    instrument_key: str
    direction: str
    quantity: int
    entry: float
    stop_loss: float
    target: float
    opened_at: str
    confidence: float
    reason: str
    last_price: float
    unrealized_pnl: float = 0.0


class PositionManager:
    """Persistent paper-only position manager. No broker execution exists."""

    def __init__(self, capital: float, settings: Settings, output_dir: Path):
        self.settings = settings
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log = self.output_dir / "paper_trades.jsonl"
        self.position_file = self.output_dir / "open_positions.json"
        self.session_file = self.output_dir / "session_state.json"
        self.starting_capital = float(capital)
        self.capital = float(capital)
        self.realized_pnl = 0.0
        self.trades_today = 0
        self.last_trade_at: str | None = None
        self.state_session_date = datetime.now(IST).date().isoformat()
        self.positions: dict[str, PaperPosition] = {}
        self._last_persist_monotonic = 0.0
        self.load_state()

    def _write_log(self, payload: dict[str, Any]) -> None:
        with self.trade_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def persist(self, force: bool = False) -> None:
        now_mono = time.monotonic()
        if not force and now_mono - self._last_persist_monotonic < self.settings.persist_interval_seconds:
            return
        self._last_persist_monotonic = now_mono
        payload = {
            "session_date": self.state_session_date,
            "capital": self.capital,
            "starting_capital": self.starting_capital,
            "realized_pnl": self.realized_pnl,
            "trades_today": self.trades_today,
            "last_trade_at": self.last_trade_at,
            "positions": [asdict(position) for position in self.positions.values()],
        }
        tmp = self.position_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.position_file)
        summary = {
            "timestamp": datetime.now(IST).isoformat(),
            "session_date": self.state_session_date,
            "capital": round(self.capital, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(sum(p.unrealized_pnl for p in self.positions.values()), 2),
            "open_positions": len(self.positions),
            "trades_today": self.trades_today,
        }
        self.session_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def load_state(self) -> None:
        if not self.position_file.exists():
            self.persist(force=True)
            return
        try:
            payload = json.loads(self.position_file.read_text(encoding="utf-8"))
            self.capital = float(payload.get("capital", self.capital))
            self.starting_capital = float(payload.get("starting_capital", self.starting_capital))
            self.realized_pnl = float(payload.get("realized_pnl", 0.0))
            self.trades_today = int(payload.get("trades_today", 0))
            self.last_trade_at = payload.get("last_trade_at")
            self.state_session_date = str(payload.get("session_date") or self.state_session_date)
            self.positions = {str(raw["symbol"]): PaperPosition(**raw) for raw in payload.get("positions", [])}
            current_date = datetime.now(IST).date().isoformat()
            if self.state_session_date != current_date:
                if self.positions:
                    print("Ignoring stale overnight paper positions from a previous session date.")
                self.positions.clear()
                self.state_session_date = current_date
                self.trades_today = 0
                self.realized_pnl = 0.0
                self.last_trade_at = None
                self.persist(force=True)
        except Exception as exc:
            print(f"Position state load skipped: {exc}")

    def restore_message(self) -> str | None:
        if not self.positions:
            return None
        lines = ["♻️ RESTORED PAPER POSITIONS", f"Open positions: {len(self.positions)}"]
        for p in self.positions.values():
            lines.append(
                f"{p.symbol} {p.direction} | Qty {p.quantity} | Entry ₹{p.entry:.2f} | Last ₹{p.last_price:.2f} | SL ₹{p.stop_loss:.2f} | Target ₹{p.target:.2f}"
            )
        return "\n".join(lines)

    def can_open(self, now: datetime) -> tuple[bool, str]:
        if len(self.positions) >= self.settings.max_trades:
            return False, "MAX_OPEN_TRADES"
        if self.trades_today >= self.settings.max_trades:
            return False, "MAX_TRADES_REACHED"
        if self.last_trade_at:
            try:
                last = datetime.fromisoformat(self.last_trade_at)
                if (now - last).total_seconds() < self.settings.cooldown_minutes * 60:
                    return False, "COOLDOWN_ACTIVE"
            except ValueError:
                pass
        return True, "OK"

    def calculate_order(self, candidate: dict[str, Any], direction: str):
        entry = float(candidate["ltp"])
        risk_amount = self.capital * self.settings.risk_per_trade_pct
        allocation_amount = self.capital * self.settings.max_allocation_pct
        if direction == "LONG":
            stop = entry * (1.0 - self.settings.stop_loss_pct)
            target = entry * (1.0 + self.settings.target_pct)
        elif direction == "SHORT":
            stop = entry * (1.0 + self.settings.stop_loss_pct)
            target = entry * (1.0 - self.settings.target_pct)
        else:
            return None
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0 or entry <= 0:
            return None
        quantity = int(min(risk_amount / risk_per_share, allocation_amount / entry))
        if quantity <= 0:
            return None
        return {"quantity": quantity, "entry": round(entry, 4), "stop_loss": round(stop, 4), "target": round(target, 4)}

    def open_position(self, candidate: dict[str, Any], decision: AITradeDecision, now: datetime):
        allowed, reason = self.can_open(now)
        if not allowed:
            return None, reason
        if decision.symbol in self.positions:
            return None, "ALREADY_OPEN"
        order = self.calculate_order(candidate, decision.direction)
        if not order:
            return None, "INVALID_POSITION_SIZE"
        position = PaperPosition(
            symbol=decision.symbol,
            instrument_key=candidate["instrument_key"],
            direction=decision.direction,
            quantity=order["quantity"],
            entry=order["entry"],
            stop_loss=order["stop_loss"],
            target=order["target"],
            opened_at=now.isoformat(),
            confidence=decision.confidence,
            reason=decision.reason,
            last_price=order["entry"],
        )
        self.positions[position.symbol] = position
        self.trades_today += 1
        self.last_trade_at = now.isoformat()
        self._write_log({"event": "OPEN", **asdict(position)})
        self.persist(force=True)
        return position, "OPENED"

    def update_ltp(self, symbol: str, ltp: float, monitor_exits: bool, timestamp: str | None = None):
        position = self.positions.get(symbol)
        if not position:
            return None
        price = float(ltp)
        position.last_price = price
        if position.direction == "LONG":
            position.unrealized_pnl = (price - position.entry) * position.quantity
            if monitor_exits and price <= position.stop_loss:
                return self.close_position(symbol, position.stop_loss, "STOP_LOSS", timestamp)
            if monitor_exits and price >= position.target:
                return self.close_position(symbol, position.target, "TARGET", timestamp)
        else:
            position.unrealized_pnl = (position.entry - price) * position.quantity
            if monitor_exits and price >= position.stop_loss:
                return self.close_position(symbol, position.stop_loss, "STOP_LOSS", timestamp)
            if monitor_exits and price <= position.target:
                return self.close_position(symbol, position.target, "TARGET", timestamp)
        self.persist(force=False)
        return None

    def close_position(self, symbol: str, exit_price: float, reason: str, timestamp: str | None = None):
        position = self.positions.pop(symbol, None)
        if not position:
            return None
        price = float(exit_price)
        pnl = ((price - position.entry) if position.direction == "LONG" else (position.entry - price)) * position.quantity
        self.realized_pnl += pnl
        self.capital += pnl
        result = {"position": position, "exit_price": price, "pnl": pnl, "reason": reason}
        self._write_log({"event": "CLOSE", **asdict(position), "exit_price": round(price, 4), "realized_pnl": round(pnl, 2), "exit_reason": reason, "closed_at": timestamp or datetime.now(IST).isoformat()})
        self.persist(force=True)
        return result

    def square_off_all(self, prices: dict[str, float], reason: str = "EOD"):
        closed = []
        for symbol in list(self.positions):
            result = self.close_position(symbol, prices.get(symbol, self.positions[symbol].last_price), reason)
            if result:
                closed.append(result)
        return closed

    def snapshot(self) -> dict[str, Any]:
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return {
            "open_count": len(self.positions),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(self.realized_pnl + unrealized, 2),
            "capital": round(self.capital, 2),
            "positions": [asdict(p) for p in self.positions.values()],
        }


class MarketHistoryCache:
    """SQLite cache for historical + live 1-minute candles; safe to cache in GitHub Actions."""

    def __init__(self, db_path: str | Path, max_bars: int = 120):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bars = max(20, int(max_bars))
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_db(self):
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("CREATE TABLE IF NOT EXISTS candles (instrument_key TEXT NOT NULL, ts TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL, PRIMARY KEY(instrument_key, ts))")
            db.execute("CREATE INDEX IF NOT EXISTS idx_candles_key_ts ON candles(instrument_key, ts)")

    @staticmethod
    def _ts_key(value: Any) -> str:
        return str(value)

    def latest_timestamp(self, instrument_key: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT ts FROM candles WHERE instrument_key=? ORDER BY ts DESC LIMIT 1", (instrument_key,)).fetchone()
        return row[0] if row else None

    def count(self, instrument_key: str) -> int:
        with self._connect() as db:
            row = db.execute("SELECT COUNT(*) FROM candles WHERE instrument_key=?", (instrument_key,)).fetchone()
        return int(row[0] or 0)

    def load(self, instrument_key: str, limit: int = 120) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT ts, open, high, low, close, volume FROM candles WHERE instrument_key=? ORDER BY ts DESC LIMIT ?", (instrument_key, int(limit))).fetchall()
        rows.reverse()
        return [{"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]} for r in rows]

    def save_many(self, instrument_key: str, bars: list[dict[str, Any]]) -> None:
        self.save_batch({instrument_key: bars})

    def save_batch(self, pending: dict[str, list[dict[str, Any]]]) -> None:
        rows = []
        keys = []
        for instrument_key, bars in pending.items():
            if not bars:
                continue
            keys.append(instrument_key)
            rows.extend(
                (instrument_key, self._ts_key(b["timestamp"]), float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"]), float(b["volume"]))
                for b in bars
            )
        if not rows:
            return
        with self._connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO candles(instrument_key, ts, open, high, low, close, volume) VALUES(?,?,?,?,?,?,?)",
                rows,
            )
            for instrument_key in set(keys):
                db.execute(
                    "DELETE FROM candles WHERE instrument_key=? AND ts NOT IN (SELECT ts FROM candles WHERE instrument_key=? ORDER BY ts DESC LIMIT ?)",
                    (instrument_key, instrument_key, self.max_bars),
                )


class HistoricalWarmup:
    """Loads enough prior 1-minute candles for all stocks without waiting for today's ticks."""

    def __init__(self, access_token: str, cache: MarketHistoryCache, settings: Settings, telegram: TelegramBot):
        self.access_token = access_token
        self.cache = cache
        self.settings = settings
        self.telegram = telegram
        self._rate_lock = asyncio.Lock()
        self._last_request_monotonic = 0.0
        self._min_interval = 60.0 / max(1, settings.history_rate_limit_per_minute)

    async def _rate_wait(self):
        async with self._rate_lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request_monotonic)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_monotonic = time.monotonic()

    @staticmethod
    def _parse_candle(raw: list[Any]) -> dict[str, Any] | None:
        if len(raw) < 6:
            return None
        return {
            "timestamp": raw[0],
            "open": float(raw[1]),
            "high": float(raw[2]),
            "low": float(raw[3]),
            "close": float(raw[4]),
            "volume": float(raw[5] or 0),
        }

    async def _fetch(self, instrument_key: str, from_date: str, to_date: str) -> list[dict[str, Any]]:
        await self._rate_wait()
        url = HISTORICAL_URL.format(instrument_key=instrument_key, to_date=to_date, from_date=from_date)
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.access_token}"}
        for attempt in range(1, 4):
            response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=25)
            if response.status_code == 200:
                payload = response.json()
                raw = payload.get("data", {}).get("candles", [])
                return [parsed for row in raw if (parsed := self._parse_candle(row))]
            if response.status_code == 429 and attempt < 3:
                await asyncio.sleep(2 * attempt)
                continue
            if response.status_code in {500, 502, 503, 504} and attempt < 3:
                await asyncio.sleep(1.5 * attempt)
                continue
            if response.status_code == 401:
                raise RuntimeError("Upstox historical-data API returned 401 Unauthorized. The access token is invalid or expired.")
            raise RuntimeError(f"Historical candle request failed for {instrument_key}: HTTP {response.status_code} - {response.text[:300]}")
        return []

    async def _fetch_intraday(self, instrument_key: str) -> list[dict[str, Any]]:
        await self._rate_wait()
        url = INTRADAY_URL.format(instrument_key=instrument_key)
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.access_token}"}
        for attempt in range(1, 4):
            response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=25)
            if response.status_code == 200:
                payload = response.json()
                raw = payload.get("data", {}).get("candles", [])
                return [parsed for row in raw if (parsed := self._parse_candle(row))]
            if response.status_code == 429 and attempt < 3:
                await asyncio.sleep(2 * attempt)
                continue
            if response.status_code in {500, 502, 503, 504} and attempt < 3:
                await asyncio.sleep(1.5 * attempt)
                continue
            if response.status_code == 401:
                raise RuntimeError("Upstox intraday candle API returned 401 Unauthorized. The access token is invalid or expired.")
            raise RuntimeError(f"Upstox intraday candle request failed for {instrument_key}: HTTP {response.status_code} - {response.text[:300]}")
        return []

    async def warm_current_session_instrument(self, instrument_key: str, scanner: DynamicStockScanner) -> int:
        bars = await self._fetch_intraday(instrument_key)
        if not bars:
            return 0
        today = datetime.now(IST).date()
        today_bars = [bar for bar in bars if scanner._to_ist_date(bar.get("timestamp")) == today]
        if not today_bars:
            return 0
        self.cache.save_many(instrument_key, today_bars)
        scanner.seed_history(instrument_key, today_bars, current_date=today)
        return len(today_bars)

    @staticmethod
    def _stale(latest: str | None, days: int) -> bool:
        if not latest:
            return True
        try:
            if latest.isdigit():
                value = int(latest)
                if value > 10_000_000_000:
                    value //= 1000
                latest_date = datetime.fromtimestamp(value, tz=IST).date()
            else:
                parsed = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=IST)
                latest_date = parsed.astimezone(IST).date()
            return (datetime.now(IST).date() - latest_date).days > days
        except Exception:
            return True

    async def warm_instruments(self, instrument_keys: list[str], scanner: DynamicStockScanner, nifty_key: str | None = None) -> None:
        if not instrument_keys:
            return
        today = datetime.now(IST).date()
        to_date = today - timedelta(days=1)
        from_date = today - timedelta(days=self.settings.history_days_back)
        targets = []
        for key in instrument_keys:
            count = self.cache.count(key)
            latest = self.cache.latest_timestamp(key)
            if count < self.settings.history_bars_per_instrument or self._stale(latest, self.settings.history_refresh_stale_days):
                targets.append(key)
            else:
                bars = self.cache.load(key, self.settings.history_bars_per_instrument)
                scanner.seed_history(key, bars, current_date=today)

        print(f"Historical cache warm: {len(instrument_keys) - len(targets)} instruments already seeded; API fetch needed for {len(targets)}.")
        if targets:
            await self.telegram.send_message(
                f"📚 Historical warm-up started. Cached: {len(instrument_keys) - len(targets)} | API fetch: {len(targets)}. No trade can be generated until the scanner is warmed."
            )

        completed = 0
        for key in targets:
            bars = await self._fetch(key, from_date.isoformat(), to_date.isoformat())
            if bars:
                self.cache.save_many(key, bars[-self.settings.history_bars_per_instrument:])
                scanner.seed_history(key, self.cache.load(key, self.settings.history_bars_per_instrument), current_date=today)
            completed += 1
            if completed % 100 == 0 or completed == len(targets):
                print(f"Historical warm-up progress: {completed}/{len(targets)}")

        if nifty_key:
            count = self.cache.count(nifty_key)
            latest = self.cache.latest_timestamp(nifty_key)
            if count < self.settings.history_bars_per_instrument or self._stale(latest, self.settings.history_refresh_stale_days):
                bars = await self._fetch(nifty_key, from_date.isoformat(), to_date.isoformat())
                if bars:
                    self.cache.save_many(nifty_key, bars[-self.settings.history_bars_per_instrument:])
            scanner.seed_nifty_history(nifty_key, self.cache.load(nifty_key, self.settings.history_bars_per_instrument), current_date=today)

        await self.telegram.send_message("✅ Historical warm-up complete. The scanner can use prior candles immediately; today's VWAP/5-minute momentum will build from live session data.")


class MinuteAggregator:
    """Build 1-minute OHLCV bars from LTPC ticks."""

    def __init__(self):
        self.current: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _minute_key(timestamp: Any) -> int:
        try:
            value = int(timestamp)
            if value > 10_000_000_000:
                value //= 1000
            return value - value % 60
        except Exception:
            current = int(time.time())
            return current - current % 60

    def update(self, instrument_key: str, ltp: float, ltq: float, timestamp: Any):
        minute = self._minute_key(timestamp)
        state = self.current.get(instrument_key)
        completed = None
        if state is None or state["minute"] != minute:
            if state is not None:
                completed = {"open": state["open"], "high": state["high"], "low": state["low"], "close": state["close"], "volume": state["volume"], "timestamp": state["minute"] * 1000}
            self.current[instrument_key] = {"minute": minute, "open": float(ltp), "high": float(ltp), "low": float(ltp), "close": float(ltp), "volume": max(float(ltq or 0), 0.0)}
        else:
            state["high"] = max(state["high"], float(ltp))
            state["low"] = min(state["low"], float(ltp))
            state["close"] = float(ltp)
            state["volume"] += max(float(ltq or 0), 0.0)
        return completed


class UpstoxLiveFeed:
    """Upstox V3 market data only. This class has no order-placement API."""

    def __init__(self, access_token: str, instrument_keys: list[str], on_tick: Callable[[str, dict[str, Any]], None], mode: str = "ltpc", reconnect_interval: int = 5, reconnect_attempts: int = 50):
        if not access_token:
            raise ValueError("UPSTOX access token is empty")
        self.access_token = access_token
        self.instrument_keys = list(dict.fromkeys(instrument_keys))
        self.on_tick = on_tick
        self.mode = mode
        self.reconnect_interval = reconnect_interval
        self.reconnect_attempts = reconnect_attempts
        self.streamer = None
        self.tick_count = 0
        self.last_tick_time = 0.0

    @staticmethod
    def _to_dict(message):
        if isinstance(message, dict):
            return message
        try:
            from google.protobuf.json_format import MessageToDict
            return MessageToDict(message, preserving_proto_field_name=True)
        except Exception:
            return None

    @staticmethod
    def _extract_ltpc(feed: dict[str, Any]):
        def walk(obj):
            if isinstance(obj, dict):
                if isinstance(obj.get("ltpc"), dict):
                    return obj["ltpc"]
                for value in obj.values():
                    found = walk(value)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for value in obj:
                    found = walk(value)
                    if found is not None:
                        return found
            return None
        return walk(feed)

    def _on_open(self):
        print(
            f"Upstox market feed connected. "
            f"Monitoring {len(self.instrument_keys)} "
            f"instruments in {self.mode} mode..."
        )

    def _on_message(self, message):
        feed = self._to_dict(message)
        if not feed:
            return
        feeds = feed.get("feeds", {})
        if not isinstance(feeds, dict):
            return
        now_ms = int(time.time() * 1000)
        for instrument_key, item in feeds.items():
            if not isinstance(item, dict):
                continue
            ltpc = self._extract_ltpc(item)
            if not ltpc or ltpc.get("ltp") is None:
                continue
            self.tick_count += 1
            self.last_tick_time = time.time()
            self.on_tick(str(instrument_key), {"ltp": float(ltpc["ltp"]), "ltq": float(ltpc.get("ltq") or 0), "timestamp": ltpc.get("ltt") or now_ms})

    def _on_error(self, error):
        print(f"Upstox feed error: {error}")

    def _on_close(self, *args):
        print(f"Upstox feed closed: {args}")

    def start_blocking(self):
        import upstox_client

        configuration = upstox_client.Configuration()
        configuration.access_token = self.access_token

        api_client = upstox_client.ApiClient(configuration)

        self.streamer = upstox_client.MarketDataStreamerV3(
            api_client,
            self.instrument_keys,
            self.mode,
        )

        self.streamer.on("open", self._on_open)
        self.streamer.on("message", self._on_message)
        self.streamer.on("error", self._on_error)
        self.streamer.on("close", self._on_close)

        self.streamer.auto_reconnect(
            True,
            self.reconnect_interval,
            self.reconnect_attempts,
        )

        self.streamer.connect()

    def stop(self):
        if self.streamer is not None:
            try:
                self.streamer.disconnect()
            except Exception:
                pass


class TradingEngine:
    """Full paper-only engine: OAuth -> historical warm-up -> live data -> scan -> Gemini -> paper positions."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        if not self.settings.paper_trading_only:
            raise RuntimeError("This project is permanently paper-only.")
        self.output_dir = Path(self.settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.telegram = TelegramBot()
        self.scanner = DynamicStockScanner(
            min_bars=self.settings.scanner_min_bars,
            min_session_bars=self.settings.scanner_min_session_bars,
            top_n=self.settings.scanner_top_n,
            min_volume_ratio=self.settings.scanner_min_volume_ratio,
            min_momentum_5m=self.settings.scanner_min_momentum_5m,
            max_rsi=self.settings.scanner_max_rsi,
            min_score=self.settings.scanner_min_score,
        )
        self.ai_selector = GeminiStockSelector(self.settings.gemini_api_key, self.settings.gemini_model, self.settings.ai_min_confidence, self.settings.ai_call_cooldown_seconds)
        self.positions = PositionManager(self.settings.starting_capital, self.settings, self.output_dir)
        self.history = MarketHistoryCache(self.settings.history_db, self.settings.history_bars_per_instrument)
        self.history_warmup = HistoricalWarmup(self.settings.upstox_access_token, self.history, self.settings, self.telegram)
        self.aggregator = MinuteAggregator()
        self.feed: UpstoxLiveFeed | None = None
        self.nifty_key: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._evaluation_task: asyncio.Task | None = None
        self._position_task: asyncio.Task | None = None
        self._session_task: asyncio.Task | None = None
        self._telegram_task: asyncio.Task | None = None
        self._feed_task: asyncio.Task | None = None
        self._evaluation_lock = asyncio.Lock()
        self._last_evaluation_mono = 0.0
        self._closed = False
        self._last_heartbeat = 0.0
        self._last_history_flush_mono = 0.0
        self._pending_history: dict[str, list[dict[str, Any]]] = {}
        self._pending_history_lock = threading.Lock()
        self._session_bootstrap_requested: set[str] = set()
        self._session_bootstrap_sem = asyncio.Semaphore(max(1, self.settings.session_bootstrap_concurrency))
        self._session_bootstrap_requests = 0
        self._session_bootstrap_lock = asyncio.Lock()
        self._last_scanner_diagnostic = 0.0

    # ------------------------- time/session -------------------------
    def now(self) -> datetime:
        return datetime.now(IST)

    def _time(self, value: str) -> dt_time:
        return Settings._clock(value)

    def after(self, clock: str, now: datetime | None = None) -> bool:
        return (now or self.now()).time() >= self._time(clock)

    def in_entry_window(self, now: datetime | None = None) -> bool:
        current = (now or self.now()).time()
        return self._time(self.settings.entry_start) <= current < self._time(self.settings.entry_end)

    def in_regular_session(self, now: datetime | None = None) -> bool:
        current = (now or self.now()).time()
        return self._time(self.settings.entry_start) <= current < self._time(self.settings.market_end)

    # ------------------------- OAuth -------------------------
    def _build_upstox_login_url(self) -> tuple[str, str]:
        state = secrets.token_urlsafe(32)
        params = {"response_type": "code", "client_id": self.settings.upstox_client_id, "redirect_uri": self.settings.upstox_redirect_uri, "state": state}
        return "https://api.upstox.com/v2/login/authorization/dialog?" + urlencode(params), state

    def _extract_upstox_code(self, message: str, expected_state: str) -> str:
        text = (message or "").strip()
        if not text:
            raise RuntimeError("Empty Upstox authentication response.")
        if text.startswith(("http://", "https://")):
            parsed = urlparse(text)
            query = parse_qs(parsed.query)
            values = query.get("code")
            returned_state = query.get("state", [""])[0]
            if not values:
                raise RuntimeError("Redirect URL does not contain an Upstox authorization code.")
            if returned_state != expected_state:
                raise RuntimeError("Upstox OAuth state validation failed.")
            return values[0]
        raise RuntimeError("Please send the full Upstox redirect URL so the OAuth state can be validated.")

    async def _exchange_upstox_code(self, authorization_code: str) -> str:
        response = await asyncio.to_thread(
            requests.post,
            "https://api.upstox.com/v2/login/authorization/token",
            headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            data={"code": authorization_code, "client_id": self.settings.upstox_client_id, "client_secret": self.settings.upstox_client_secret, "redirect_uri": self.settings.upstox_redirect_uri, "grant_type": "authorization_code"},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Upstox token exchange failed: HTTP {response.status_code} - {response.text[:500]}")
        token = response.json().get("access_token")
        if not token:
            raise RuntimeError("Upstox token response did not contain access_token")
        return token

    async def _authenticate_upstox(self) -> str:
        if not self.settings.upstox_oauth_required:
            if not self.settings.upstox_access_token:
                raise RuntimeError("UPSTOX_OAUTH_REQUIRED=false but no UPSTOX_ACCESS_TOKEN was provided.")
            print("Using supplied Upstox access token; OAuth skipped.")
            return self.settings.upstox_access_token

        login_url, state = self._build_upstox_login_url()
        await self.telegram.send_message("🔐 Upstox login required. I will send the login URL next. Open it, complete login, then send the FULL redirect URL back to this Telegram bot. Execution remains PAPER ONLY.")
        response_text = await self.telegram.send_link_and_wait_for_code(login_url, timeout=300)
        if not response_text:
            raise RuntimeError("No Upstox authorization response was received within the timeout.")
        authorization_code = self._extract_upstox_code(response_text, state)
        print("Upstox authorization code received; exchanging for access token...")
        access_token = await self._exchange_upstox_code(authorization_code)
        self.settings.upstox_access_token = access_token
        token_path = Path(self.settings.upstox_access_token_file)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(access_token, encoding="utf-8")
        print("Upstox access token generated and saved.")
        await self.telegram.send_message("✅ Upstox authentication successful. Access token saved. Starting market-data authorization. Execution: PAPER ONLY.")
        return access_token

    async def _preflight_market_feed_auth(self, access_token: str) -> None:
        response = await asyncio.to_thread(
            requests.get,
            "https://api.upstox.com/v3/feed/market-data-feed/authorize",
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            timeout=20,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Upstox V3 market-feed authorization failed: HTTP {response.status_code} - {response.text[:500]}")
        payload = response.json()
        if not payload.get("data", {}).get("authorized_redirect_uri"):
            raise RuntimeError(f"Upstox feed authorization returned no authorized_redirect_uri: {payload}")
        print("Upstox V3 market-feed authorization preflight: OK")

    # ------------------------- startup / historical warm-up -------------------------
    def prepare_universe(self):
        path = Path(self.settings.instrument_file)
        if self.settings.auto_download_instruments:
            try:
                print(f"Downloading current Upstox NSE instrument master -> {path}")
                DynamicStockScanner.download_instrument_file(path, self.settings.instrument_url)
            except Exception as exc:
                if not path.exists():
                    raise RuntimeError(f"Could not download NSE instrument file: {exc}") from exc
                print(f"Instrument download failed; using existing file: {exc}")
        instruments = DynamicStockScanner.load_upstox_instruments(path)
        if len(instruments) + 1 > self.settings.max_subscribed_instruments:
            raise RuntimeError(f"Universe {len(instruments)} + NIFTY exceeds the configured subscription cap {self.settings.max_subscribed_instruments}.")
        self.scanner.register_instruments(instruments.values())
        self.nifty_key = DynamicStockScanner.find_nifty50_key(path)
        if self.nifty_key:
            print(f"Resolved NIFTY 50 instrument key: {self.nifty_key}")
        else:
            print("WARNING: NIFTY 50 instrument key was not resolved; relative-strength context will be unavailable.")
        print(f"Registered {len(instruments)} NSE_EQ instruments.")

    async def warm_history(self):
        stock_keys = list(self.scanner.states.keys())
        warmup = HistoricalWarmup(self.settings.upstox_access_token, self.history, self.settings, self.telegram)
        await warmup.warm_instruments(stock_keys, self.scanner, self.nifty_key)

    async def start(self, start_telegram_polling: bool = True):
        self._loop = asyncio.get_running_loop()
        if not start_telegram_polling:
            raise RuntimeError("Telegram polling is required for the daily Upstox OAuth flow.")
        self._telegram_task = asyncio.create_task(self.telegram.start())
        await asyncio.sleep(1)

        access_token = await self._authenticate_upstox()
        self.history_warmup.access_token = access_token
        await self._preflight_market_feed_auth(access_token)
        self.prepare_universe()
        await self.warm_history()

        restored = self.positions.restore_message()
        await self.telegram.send_message(
            "🤖 AI PAPER TRADING ENGINE READY\n"
            "Execution: PAPER ONLY\n"
            f"NSE stocks: {len(self.scanner.states)}\n"
            f"NIFTY context: {'READY' if self.nifty_key else 'UNAVAILABLE'}\n"
            f"Entry: {self.settings.entry_start}-{self.settings.entry_end} IST\n"
            f"Square-off: {self.settings.square_off} IST\n"
            f"Market end: {self.settings.market_end} IST\n"
            f"Gemini: {self.settings.gemini_model}\n"
            "Historical indicators are pre-warmed; current-session VWAP and momentum build from live data."
        )
        if restored:
            await self.telegram.send_message(restored)

        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        self._position_task = asyncio.create_task(self._position_loop())
        self._session_task = asyncio.create_task(self._session_loop())
        if self.settings.live_feed_enabled:
            self._feed_task = asyncio.create_task(self._start_feed())

    async def _start_feed(self):
        try:
            keys = list(self.scanner.states.keys())
            if self.nifty_key:
                keys.append(self.nifty_key)
            self.feed = UpstoxLiveFeed(self.settings.upstox_access_token, keys, self._on_live_tick, self.settings.upstox_mode, self.settings.reconnect_interval_seconds, self.settings.reconnect_attempts)
            await asyncio.to_thread(self.feed.start_blocking)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Upstox market feed failed: {exc}")
            await self.telegram.send_message(f"❌ Upstox market feed failed: {exc}")

    def _queue_history(self, instrument_key: str, bar: dict[str, Any]) -> None:
        with self._pending_history_lock:
            self._pending_history.setdefault(instrument_key, []).append(bar)

    def _flush_pending_history(self) -> None:
        with self._pending_history_lock:
            if not self._pending_history:
                return
            pending = self._pending_history
            self._pending_history = {}
        self.history.save_batch(pending)

    async def _bootstrap_current_session(self, instrument_key: str) -> None:
        if not self.settings.session_bootstrap_enabled:
            return
        if instrument_key == self.nifty_key:
            return

        async with self._session_bootstrap_lock:
            if instrument_key in self._session_bootstrap_requested:
                return
            if self._session_bootstrap_requests >= self.settings.session_bootstrap_max_requests:
                return
            self._session_bootstrap_requested.add(instrument_key)
            self._session_bootstrap_requests += 1

        try:
            async with self._session_bootstrap_sem:
                count = await self.history_warmup.warm_current_session_instrument(instrument_key, self.scanner)
                if count >= self.settings.session_bootstrap_min_bars:
                    state = self.scanner.states.get(instrument_key)
                    symbol = state.instrument.trading_symbol if state else instrument_key.split("|")[-1]
                    print(f"Session bootstrap | {symbol} | bars={count}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Session bootstrap failed | {instrument_key}: {exc}")

    # ------------------------- live data -------------------------
    def _on_live_tick(self, instrument_key: str, tick: dict[str, Any]):
        price = float(tick["ltp"])
        timestamp = tick.get("timestamp")
        if self.nifty_key and instrument_key == self.nifty_key:
            completed = self.aggregator.update(instrument_key, price, float(tick.get("ltq") or 0), timestamp)
            if completed:
                self.scanner.ingest_index_bar(instrument_key, completed)
                self._queue_history(instrument_key, completed)
            return

        state = self.scanner.states.get(instrument_key)
        symbol = state.instrument.trading_symbol if state else instrument_key.split("|")[-1]
        if (
            self.settings.session_bootstrap_enabled
            and state is not None
            and len(state.session_bars) < self.settings.session_bootstrap_min_bars
            and self.in_regular_session()
        ):
            self._schedule(self._bootstrap_current_session(instrument_key))

        closed = self.positions.update_ltp(symbol, price, monitor_exits=self.in_regular_session(), timestamp=str(timestamp) if timestamp else None)
        if closed:
            self._schedule(self._notify_exit(closed))
        completed = self.aggregator.update(instrument_key, price, float(tick.get("ltq") or 0), timestamp)
        if completed:
            self.scanner.ingest_bar(instrument_key, completed, trading_symbol=symbol)
            self._queue_history(instrument_key, completed)

        now = time.time()
        if now - self._last_heartbeat >= 60:
            self._last_heartbeat = now
            print(f"Live feed heartbeat | ticks={self.feed.tick_count if self.feed else 0} | last={symbol} ₹{price:.2f}")

    # Compatibility hooks for an existing Data_Bus / streamer.
    def on_bar(self, instrument_key: str, bar: dict[str, Any], trading_symbol: str | None = None, sector: str = "") -> bool:
        accepted = self.scanner.ingest_bar(instrument_key, bar, trading_symbol=trading_symbol, sector=sector)
        if accepted:
            self._queue_history(instrument_key, bar)
        return accepted

    def on_upstox_feed(self, instrument_key: str, feed: dict[str, Any], trading_symbol: str | None = None, sector: str = "") -> bool:
        accepted = self.scanner.ingest_upstox_feed(instrument_key, feed, trading_symbol=trading_symbol, sector=sector) if hasattr(self.scanner, "ingest_upstox_feed") else False
        return accepted

    def update_market_context(self, context: dict[str, Any]):
        self.scanner.update_market_context(context)

    # ------------------------- loops -------------------------
    async def _evaluation_loop(self):
        while not self._closed:
            try:
                await asyncio.sleep(max(1, self.settings.evaluate_every_seconds))
                if self.in_entry_window():
                    await self.evaluate()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"Evaluation loop error: {exc}")
                await self.telegram.send_message(f"⚠️ Evaluation error: {exc}")

    async def _position_loop(self):
        last_snapshot_mono = 0.0

        while not self._closed:
            try:
                await asyncio.sleep(1)
                now = self.now()

                if (
                        self.after(self.settings.square_off, now)
                        and self.in_regular_session(now)
                        and self.positions.positions
                ):
                    prices = {
                        symbol: p.last_price
                        for symbol, p in self.positions.positions.items()
                    }

                    closed = self.positions.square_off_all(
                        prices,
                        "EOD_SQUARE_OFF"
                    )

                    for item in closed:
                        await self._notify_exit(item)

                if (
                        self.positions.positions
                        and self.settings.position_snapshot_seconds > 0
                        and time.monotonic() - last_snapshot_mono >= self.settings.position_snapshot_seconds
                ):
                    last_snapshot_mono = time.monotonic()

                    await self.telegram.send_message(
                        self._position_snapshot_text()
                    )

                self.positions.persist(force=False)

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                print(f"Position loop error: {exc}")

    async def _session_loop(self):
        while not self._closed:
            await asyncio.sleep(1)
            now = self.now()
            stop = self.settings.run_stop_time or self.settings.market_end
            if now.time() >= self._time(stop):
                if self.positions.positions:
                    prices = {symbol: p.last_price for symbol, p in self.positions.positions.items()}
                    closed = self.positions.square_off_all(prices, "SESSION_END")
                    for item in closed:
                        await self._notify_exit(item)
                await self.telegram.send_message(self._daily_summary())
                self.positions.persist(force=True)
                await self.close()
                return

    # ------------------------- AI / paper entry -------------------------
    async def evaluate(self, force: bool = False):
        if self.positions.positions and len(self.positions.positions) >= self.settings.max_trades:
            return None
        allowed, _ = self.positions.can_open(self.now())
        if not allowed and not force:
            return None
        now_mono = time.monotonic()
        if not force and now_mono - self._last_evaluation_mono < self.settings.evaluate_every_seconds:
            return None
        if self._evaluation_lock.locked():
            return None
        if not force and not self.in_entry_window():
            return None

        async with self._evaluation_lock:
            self._last_evaluation_mono = time.monotonic()
            candidates = self.scanner.scan()
            now_diag = time.monotonic()
            if now_diag - self._last_scanner_diagnostic >= 30:
                self._last_scanner_diagnostic = now_diag
                print(self.scanner.diagnostic_summary())
            if not candidates:
                print("Scanner: no qualifying candidates.")
                return None
            print(self.scanner.candidate_summary(candidates))
            decision = await self.ai_selector.select(candidates, self.scanner.market_context, "LONG_SHORT" if self.settings.allow_short else "LONG_ONLY")
            await self._log_decision(candidates, decision)
            if self.settings.telegram_scan_messages:
                await self._notify_scan(candidates, decision)
            if decision.decision != "TRADE":
                print(f"Gemini decision: NO_TRADE | {decision.reason}")
                return decision
            candidate = next((item for item in candidates if item["symbol"] == decision.symbol), None)
            if candidate is None:
                return decision
            position, status = self.positions.open_position(candidate, decision, self.now())
            if position:
                await self.telegram.send_message(
                    f"🟢 PAPER ENTRY\nSymbol: {position.symbol}\nDirection: {position.direction}\nQty: {position.quantity}\nEntry: ₹{position.entry:.2f}\nStop: ₹{position.stop_loss:.2f}\nTarget: ₹{position.target:.2f}\nAI confidence: {position.confidence:.2f}\nReason: {position.reason}"
                )
            else:
                await self.telegram.send_message(f"⚠️ Paper trade skipped: {decision.symbol} | {status}")
            return decision

    # ------------------------- notifications/logging -------------------------
    async def _notify_exit(self, result: dict[str, Any]):
        position = result["position"]
        pnl = result["pnl"]
        emoji = "✅" if pnl >= 0 else "🔴"
        await self.telegram.send_message(
            f"{emoji} PAPER EXIT\nSymbol: {position.symbol}\nDirection: {position.direction}\nQty: {position.quantity}\nEntry: ₹{position.entry:.2f}\nExit: ₹{result['exit_price']:.2f}\nReason: {result['reason']}\nP&L: ₹{pnl:.2f}\nRealized P&L: ₹{self.positions.realized_pnl:.2f}"
        )

    def _position_snapshot_text(self) -> str:
        snap = self.positions.snapshot()
        lines = [f"📊 OPEN PAPER POSITIONS\nRealized P&L: ₹{snap['realized_pnl']:.2f}\nUnrealized P&L: ₹{snap['unrealized_pnl']:.2f}\nTotal P&L: ₹{snap['total_pnl']:.2f}"]
        for position in snap["positions"]:
            lines.append(f"{position['symbol']} {position['direction']} | Qty {position['quantity']} | LTP ₹{position['last_price']:.2f} | P&L ₹{position['unrealized_pnl']:.2f}")
        return "\n".join(lines)

    def _daily_summary(self) -> str:
        snap = self.positions.snapshot()
        return f"🏁 PAPER SESSION COMPLETE\nRealized P&L: ₹{snap['realized_pnl']:.2f}\nUnrealized P&L: ₹{snap['unrealized_pnl']:.2f}\nTotal P&L: ₹{snap['total_pnl']:.2f}\nCapital: ₹{snap['capital']:.2f}\nLive order API: DISABLED"

    async def _notify_scan(self, candidates, decision):
        top = [f"{x['symbol']} score={x['scanner_score']:.0f} Vol={x['volume_ratio']:.1f}x RSI={x['rsi']:.0f} RS={x['relative_strength']:.2f}%" for x in candidates[:10]]
        await self.telegram.send_message("🔎 DYNAMIC SCAN\n" + "\n".join(top) + f"\n\nAI: {decision.decision} {decision.symbol} {decision.direction}\nConfidence: {decision.confidence:.2f}\nReason: {decision.reason}")

    async def _log_decision(self, candidates, decision):
        payload = {"timestamp": self.now().isoformat(), "market_context": self.scanner.market_context, "candidates": candidates, "decision": decision.model_dump()}
        with (self.output_dir / "ai_decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _schedule(self, coroutine):
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._flush_pending_history)
        self.positions.persist(force=True)
        if self.feed:
            self.feed.stop()
        current = asyncio.current_task()
        tasks = (self._evaluation_task, self._position_task, self._session_task, self._feed_task, self._telegram_task)
        for task in tasks:
            if task and task is not current:
                task.cancel()
        for task in tasks:
            if task and task is not current:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
        await self.telegram.close()


async def main():
    engine = TradingEngine()
    await engine.start(start_telegram_polling=True)
    print("Running: historical warm-up -> Upstox live feed -> dynamic scanner -> Gemini -> PAPER positions")
    print("IMPORTANT: This project contains NO live order-placement code.")
    try:
        while not engine._closed:
            await asyncio.sleep(1)
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
