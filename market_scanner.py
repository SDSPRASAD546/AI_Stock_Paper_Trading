from __future__ import annotations

import gzip
import json
import math
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"


@dataclass
class InstrumentInfo:
    instrument_key: str
    trading_symbol: str
    name: str = ""
    sector: str = ""


@dataclass
class StockState:
    instrument: InstrumentInfo
    bars: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=120))
    session_bars: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=400))
    last_bar_timestamp: str | int | None = None
    session_date: str | None = None


class DynamicStockScanner:
    """Dynamic NSE scanner with historical warm-up and current-session metrics."""

    def __init__(
        self,
        min_bars: int = 20,
        min_session_bars: int = 5,
        top_n: int = 15,
        min_price: float = 50.0,
        max_price: float = 100000.0,
        min_volume_ratio: float = 1.10,
        max_rsi: float = 72.0,
        min_momentum_5m: float = 0.10,
        min_score: float = 45.0,
    ):
        self.min_bars = int(min_bars)
        self.min_session_bars = int(min_session_bars)
        self.top_n = int(top_n)
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.min_volume_ratio = float(min_volume_ratio)
        self.max_rsi = float(max_rsi)
        self.min_momentum_5m = float(min_momentum_5m)
        self.min_score = float(min_score)

        self.states: dict[str, StockState] = {}
        self.market_context: dict[str, Any] = {}
        self.nifty_instrument_key: str | None = None
        self.nifty_bars: deque[dict[str, Any]] = deque(maxlen=120)
        self.nifty_session_bars: deque[dict[str, Any]] = deque(maxlen=400)
        self._last_scan_diagnostics: dict[str, int] = {}

    # ------------------------- universe -------------------------
    @staticmethod
    def download_instrument_file(path: str | Path, url: str = DEFAULT_INSTRUMENT_URL, timeout: int = 30) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        request = urllib.request.Request(url, headers={"User-Agent": "AITradingBot/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        if not data:
            raise RuntimeError("Upstox instrument download returned an empty file")
        tmp.write_bytes(data)
        tmp.replace(destination)
        return destination

    @staticmethod
    def _load_raw_instruments(path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Instrument file not found: {path}")
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                raw = json.load(handle)
        else:
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        if isinstance(raw, dict):
            raw = raw.get("data", raw.get("instruments", raw))
        if not isinstance(raw, list):
            raise ValueError("Unsupported Upstox instrument-file structure")
        return [item for item in raw if isinstance(item, dict)]

    @classmethod
    def load_upstox_instruments(cls, path: str | Path) -> dict[str, InstrumentInfo]:
        result: dict[str, InstrumentInfo] = {}
        for item in cls._load_raw_instruments(path):
            if item.get("segment") != "NSE_EQ" or item.get("instrument_type") != "EQ":
                continue
            key = item.get("instrument_key")
            symbol = item.get("trading_symbol")
            if not key or not symbol:
                continue
            result[str(key)] = InstrumentInfo(
                instrument_key=str(key),
                trading_symbol=str(symbol),
                name=str(item.get("name") or item.get("short_name") or ""),
                sector=str(item.get("sector") or ""),
            )
        return result

    @classmethod
    def find_nifty50_key(cls, path: str | Path) -> str | None:
        for item in cls._load_raw_instruments(path):
            segment = str(item.get("segment") or "")
            symbol = str(item.get("trading_symbol") or "").strip().upper()
            key = str(item.get("instrument_key") or "")
            if not key or not segment.startswith("NSE_INDEX"):
                continue
            if symbol in {"NIFTY 50", "NIFTY50", "NIFTY"}:
                return key
            if key.upper() == "NSE_INDEX|NIFTY 50":
                return key
        return None

    def register_instruments(self, instruments: Iterable[InstrumentInfo]) -> None:
        for instrument in instruments:
            self.states.setdefault(instrument.instrument_key, StockState(instrument=instrument))

    def register_instrument(self, instrument_key: str, trading_symbol: str, name: str = "", sector: str = "") -> None:
        self.states.setdefault(
            instrument_key,
            StockState(instrument=InstrumentInfo(instrument_key=instrument_key, trading_symbol=trading_symbol, name=name, sector=sector)),
        )

    # ------------------------- time helpers -------------------------
    @staticmethod
    def _to_ist_date(timestamp: Any) -> date | None:
        if timestamp is None:
            return None
        try:
            if isinstance(timestamp, (int, float)):
                value = int(timestamp)
                if value > 10_000_000_000:
                    value //= 1000
                return datetime.fromtimestamp(value, tz=IST).date()
            text = str(timestamp)
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=IST)
            return parsed.astimezone(IST).date()
        except Exception:
            return None

    # ------------------------- data -------------------------
    def seed_history(self, instrument_key: str, bars: Iterable[dict[str, Any]], current_date: date | None = None) -> int:
        state = self.states.get(instrument_key)
        if state is None:
            return 0
        if current_date is None:
            current_date = datetime.now(IST).date()

        existing_history = {str(item.get("timestamp")) for item in state.bars}
        if state.session_date != current_date.isoformat():
            state.session_bars.clear()
            state.session_date = current_date.isoformat()
        existing_session = {str(item.get("timestamp")) for item in state.session_bars}

        count = 0
        for bar in bars:
            try:
                clean = {
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": max(float(bar.get("volume", 0.0)), 0.0),
                    "timestamp": bar.get("timestamp"),
                }
            except (TypeError, ValueError, KeyError):
                continue

            ts_key = str(clean["timestamp"])
            if ts_key not in existing_history:
                state.bars.append(clean)
                existing_history.add(ts_key)
                count += 1
            else:
                # Refresh an existing candle in place when a live/intraday source
                # provides a newer version of the same timestamp.
                for index in range(len(state.bars) - 1, -1, -1):
                    if str(state.bars[index].get("timestamp")) == ts_key:
                        state.bars[index] = clean
                        break

            state.last_bar_timestamp = clean["timestamp"]
            bar_date = self._to_ist_date(clean["timestamp"])
            if bar_date == current_date and ts_key not in existing_session:
                state.session_bars.append(clean)
                existing_session.add(ts_key)
            elif bar_date == current_date:
                for index in range(len(state.session_bars) - 1, -1, -1):
                    if str(state.session_bars[index].get("timestamp")) == ts_key:
                        state.session_bars[index] = clean
                        break
        return count

    def seed_nifty_history(self, instrument_key: str, bars: Iterable[dict[str, Any]], current_date: date | None = None) -> int:
        self.nifty_instrument_key = instrument_key
        if current_date is None:
            current_date = datetime.now(IST).date()
        count = 0
        for bar in bars:
            clean = {
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": max(float(bar.get("volume", 0.0)), 0.0),
                "timestamp": bar.get("timestamp"),
            }
            self.nifty_bars.append(clean)
            bar_date = self._to_ist_date(clean["timestamp"])
            if bar_date == current_date:
                self.nifty_session_bars.append(clean)
            count += 1
        self._update_nifty_context()
        return count

    def ingest_bar(self, instrument_key: str, bar: dict[str, Any], trading_symbol: str | None = None, sector: str = "") -> bool:
        if instrument_key not in self.states:
            self.register_instrument(instrument_key, trading_symbol or instrument_key.split("|")[-1], sector=sector)
        required = ("open", "high", "low", "close", "volume")
        if any(key not in bar for key in required):
            raise ValueError(f"Bar missing required fields: {required}")

        clean = {
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": max(float(bar["volume"]), 0.0),
            "timestamp": bar.get("timestamp"),
        }
        state = self.states[instrument_key]
        ts_key = str(clean["timestamp"])

        # Replace an already-seen minute rather than dropping it. This matters when
        # an intraday bootstrap provides a partial/current candle and the live feed
        # later provides the completed version.
        if state.last_bar_timestamp == clean["timestamp"]:
            if state.bars and str(state.bars[-1].get("timestamp")) == ts_key:
                state.bars[-1] = clean
            if state.session_bars and str(state.session_bars[-1].get("timestamp")) == ts_key:
                state.session_bars[-1] = clean
            return True

        state.bars.append(clean)
        state.last_bar_timestamp = clean["timestamp"]
        bar_date = self._to_ist_date(clean["timestamp"])
        current_date = datetime.now(IST).date()
        if bar_date == current_date:
            date_key = current_date.isoformat()
            if state.session_date != date_key:
                state.session_bars.clear()
                state.session_date = date_key
            state.session_bars.append(clean)
        return True

    def ingest_index_bar(self, instrument_key: str, bar: dict[str, Any]) -> bool:
        self.nifty_instrument_key = instrument_key
        clean = {
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": max(float(bar.get("volume", 0.0)), 0.0),
            "timestamp": bar.get("timestamp"),
        }
        self.nifty_bars.append(clean)
        bar_date = self._to_ist_date(clean["timestamp"])
        current_date = datetime.now(IST).date()
        if bar_date == current_date:
            self.nifty_session_bars.append(clean)
        self._update_nifty_context()
        return True

    def ingest_upstox_feed(self, instrument_key: str, feed: dict[str, Any], trading_symbol: str | None = None, sector: str = "") -> bool:
        """Compatibility hook for an existing MessageToDict-style Upstox feed containing marketOHLC/I1."""
        def find_key(obj: Any, key: str):
            if isinstance(obj, dict):
                if key in obj:
                    return obj[key]
                for value in obj.values():
                    found = find_key(value, key)
                    if found is not None:
                        return found
            elif isinstance(obj, list):
                for value in obj:
                    found = find_key(value, key)
                    if found is not None:
                        return found
            return None

        market_ohlc = find_key(feed, "marketOHLC")
        if not isinstance(market_ohlc, dict):
            return False
        candles = market_ohlc.get("ohlc", [])
        if not isinstance(candles, list):
            return False
        candle = next((item for item in reversed(candles) if str(item.get("interval", "")) in {"I1", "1m", "1minute"}), None)
        if not candle:
            return False
        return self.ingest_bar(
            instrument_key,
            {
                "open": candle.get("open"),
                "high": candle.get("high"),
                "low": candle.get("low"),
                "close": candle.get("close"),
                "volume": candle.get("vol", candle.get("volume", 0)),
                "timestamp": candle.get("ts"),
            },
            trading_symbol=trading_symbol,
            sector=sector,
        )

    # ------------------------- indicators -------------------------
    @staticmethod
    def _ema(values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        alpha = 2.0 / (period + 1.0)
        ema = sum(values[:period]) / period
        for value in values[period:]:
            ema = alpha * value + (1.0 - alpha) * ema
        return ema

    @staticmethod
    def _rsi(values: list[float], period: int = 14) -> float | None:
        if len(values) < period + 1:
            return None
        gains: list[float] = []
        losses: list[float] = []
        for previous, current in zip(values[-period - 1:-1], values[-period:]):
            change = current - previous
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    @staticmethod
    def _vwap(session_bars: list[dict[str, Any]]) -> float | None:
        if not session_bars:
            return None
        numerator = 0.0
        denominator = 0.0
        for bar in session_bars:
            typical = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            volume = max(float(bar.get("volume", 0.0)), 0.0)
            numerator += typical * volume
            denominator += volume
        return numerator / denominator if denominator > 0 else None

    @staticmethod
    def _volume_ratio(bars: list[dict[str, Any]], lookback: int = 20) -> float:
        if len(bars) < lookback + 1:
            return 0.0
        current = max(float(bars[-1]["volume"]), 0.0)
        baseline = [max(float(bar["volume"]), 0.0) for bar in bars[-lookback - 1:-1]]
        average = sum(baseline) / len(baseline) if baseline else 0.0
        return current / average if average > 0 else 0.0

    @staticmethod
    def _momentum_pct(bars: list[dict[str, Any]], periods: int = 5) -> float:
        if len(bars) <= periods:
            return 0.0
        previous = float(bars[-periods - 1]["close"])
        current = float(bars[-1]["close"])
        return (current / previous - 1.0) * 100.0 if previous else 0.0

    def _update_nifty_context(self) -> None:
        momentum = self._momentum_pct(list(self.nifty_session_bars), 5)
        close = self.nifty_session_bars[-1]["close"] if self.nifty_session_bars else (self.nifty_bars[-1]["close"] if self.nifty_bars else 0.0)
        self.market_context.update(
            {
                "nifty_momentum_5m": round(momentum, 4),
                "nifty_ltp": round(float(close), 2),
                "nifty_ready": len(self.nifty_session_bars) >= 6,
            }
        )

    # ------------------------- scoring -------------------------
    @staticmethod
    def _score_long(close: float, vwap: float | None, ema9: float | None, ema20: float | None, rsi: float | None, volume_ratio: float, momentum_5m: float, relative_strength: float) -> float:
        score = 0.0
        if vwap is not None and close > vwap:
            score += 20.0
        if ema9 is not None and ema20 is not None and ema9 > ema20:
            score += 15.0
        if volume_ratio >= 1.5:
            score += 20.0
        elif volume_ratio >= 1.2:
            score += 12.0
        elif volume_ratio >= 1.1:
            score += 6.0
        if momentum_5m >= 0.50:
            score += 20.0
        elif momentum_5m >= 0.25:
            score += 14.0
        elif momentum_5m >= 0.10:
            score += 8.0
        if relative_strength > 0.50:
            score += 15.0
        elif relative_strength > 0:
            score += 8.0
        if rsi is not None:
            if 55 <= rsi <= 68:
                score += 10.0
            elif 50 <= rsi < 55 or 68 < rsi <= 72:
                score += 5.0
        return score

    # ------------------------- scan -------------------------
    def scan(self) -> list[dict[str, Any]]:
        nifty_momentum = float(self.market_context.get("nifty_momentum_5m", 0.0) or 0.0)
        candidates: list[dict[str, Any]] = []
        stats = {
            "total": len(self.states),
            "not_enough_history": 0,
            "not_enough_session": 0,
            "price": 0,
            "volume": 0,
            "momentum": 0,
            "rsi": 0,
            "score": 0,
            "qualified": 0,
        }

        for state in self.states.values():
            bars = list(state.bars)
            session_bars = list(state.session_bars)
            if len(bars) < self.min_bars:
                stats["not_enough_history"] += 1
                continue
            if len(session_bars) < self.min_session_bars:
                stats["not_enough_session"] += 1
                continue

            close = float(bars[-1]["close"])
            if not self.min_price <= close <= self.max_price:
                stats["price"] += 1
                continue

            closes = [float(bar["close"]) for bar in bars]
            ema9 = self._ema(closes, 9)
            ema20 = self._ema(closes, 20)
            rsi = self._rsi(closes, 14)
            vwap = self._vwap(session_bars)
            volume_ratio = self._volume_ratio(bars, 20)
            momentum_5m = self._momentum_pct(session_bars, 5)
            session_open = float(session_bars[0]["open"])
            change_pct = (close / session_open - 1.0) * 100.0 if session_open else 0.0
            relative_strength = momentum_5m - nifty_momentum

            if volume_ratio < self.min_volume_ratio:
                stats["volume"] += 1
                continue
            if momentum_5m < self.min_momentum_5m:
                stats["momentum"] += 1
                continue
            if rsi is None or rsi > self.max_rsi:
                stats["rsi"] += 1
                continue

            score = self._score_long(close, vwap, ema9, ema20, rsi, volume_ratio, momentum_5m, relative_strength)
            if score < self.min_score:
                stats["score"] += 1
                continue

            stats["qualified"] += 1
            candidates.append(
                {
                    "instrument_key": state.instrument.instrument_key,
                    "symbol": state.instrument.trading_symbol,
                    "name": state.instrument.name,
                    "sector": state.instrument.sector,
                    "ltp": round(close, 4),
                    "change_pct": round(change_pct, 4),
                    "above_vwap": bool(vwap is not None and close > vwap),
                    "vwap_distance_pct": round(((close - vwap) / vwap * 100.0) if vwap else 0.0, 4),
                    "ema9": round(ema9, 4) if ema9 is not None else None,
                    "ema20": round(ema20, 4) if ema20 is not None else None,
                    "rsi": round(rsi, 2) if rsi is not None else None,
                    "volume_ratio": round(volume_ratio, 3),
                    "momentum_5m": round(momentum_5m, 4),
                    "nifty_momentum_5m": round(nifty_momentum, 4),
                    "relative_strength": round(relative_strength, 4),
                    "scanner_score": round(score, 2),
                    "history_bars": len(bars),
                    "session_bars": len(session_bars),
                }
            )

        candidates.sort(key=lambda item: item["scanner_score"], reverse=True)
        self._last_scan_diagnostics = stats
        return candidates[: self.top_n]

    def diagnostic_summary(self) -> str:
        s = self._last_scan_diagnostics
        if not s:
            return "Scanner diagnostics | no scan performed yet."
        return (
            "Scanner diagnostics | "
            f"Total={s.get('total', 0)} | "
            f"History={s.get('not_enough_history', 0)} | "
            f"Session={s.get('not_enough_session', 0)} | "
            f"Price={s.get('price', 0)} | "
            f"Volume={s.get('volume', 0)} | "
            f"Momentum={s.get('momentum', 0)} | "
            f"RSI={s.get('rsi', 0)} | "
            f"Score={s.get('score', 0)} | "
            f"Qualified={s.get('qualified', 0)}"
        )

    def candidate_summary(self, candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return "No qualifying dynamic candidates."
        lines = ["Top dynamic candidates:"]
        for item in candidates:
            lines.append(
                f"{item['symbol']} | score={item['scanner_score']:.1f} "
                f"RS={item['relative_strength']:.2f}% "
                f"Vol={item['volume_ratio']:.2f}x "
                f"RSI={item['rsi']:.1f} "
                f"VWAP={'YES' if item['above_vwap'] else 'NO'}"
            )
        return "\n".join(lines)
