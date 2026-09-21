from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError


class AITradeDecision(BaseModel):
    decision: Literal["TRADE", "NO_TRADE"]
    symbol: str
    direction: Literal["LONG", "SHORT", "NONE"]
    confidence: float = Field(ge=0.0, le=1.0)
    setup: str
    reason: str
    risk_flags: list[str] = Field(default_factory=list)
    candidates_considered: list[str] = Field(default_factory=list)


class GeminiStockSelector:
    """Gemini may select only from the dynamically generated candidate list."""

    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite", min_confidence: float = 0.70, min_interval_seconds: int = 60):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.min_confidence = float(min_confidence)
        self.min_interval_seconds = max(0, int(min_interval_seconds))
        self._last_call_monotonic = 0.0

    def _prompt(self, candidates: list[dict[str, Any]], market_context: dict[str, Any], direction_mode: str) -> str:
        compact = [
            {key: item.get(key) for key in (
                "symbol", "sector", "ltp", "change_pct", "above_vwap", "vwap_distance_pct",
                "ema9", "ema20", "rsi", "volume_ratio", "momentum_5m",
                "nifty_momentum_5m", "relative_strength", "scanner_score", "history_bars", "session_bars"
            )}
            for item in candidates
        ]
        direction_rule = "LONG only. Do not return SHORT." if direction_mode == "LONG_ONLY" else "LONG or SHORT only when the supplied data supports it."
        return (
            "You are the stock-selection module inside an intraday paper-trading system.\n\n"
            "Rules:\n"
            "- The candidate universe is dynamic and changes every scan.\n"
            "- Select ONLY a symbol present in the supplied candidate list.\n"
            "- Never invent prices, indicators, volume, news, or market conditions.\n"
            "- Use only supplied data.\n"
            "- Return NO_TRADE when evidence is weak, conflicting, or incomplete.\n"
            f"- {direction_rule}\n"
            "- Do not calculate position size or place orders.\n\n"
            "Prioritize multi-indicator agreement, momentum + volume, VWAP alignment, and relative strength.\n\n"
            f"Market context:\n{json.dumps(market_context, ensure_ascii=False, indent=2)}\n\n"
            f"Dynamic candidates:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
            "For NO_TRADE use symbol=NONE and direction=NONE."
        )

    @staticmethod
    def _no_trade(reason: str, flags: list[str], candidates: list[dict[str, Any]]) -> AITradeDecision:
        return AITradeDecision(
            decision="NO_TRADE",
            symbol="NONE",
            direction="NONE",
            confidence=0.0,
            setup="NO_TRADE",
            reason=reason,
            risk_flags=flags,
            candidates_considered=[item["symbol"] for item in candidates],
        )

    async def select(self, candidates: list[dict[str, Any]], market_context: dict[str, Any], direction_mode: str = "LONG_ONLY") -> AITradeDecision:
        if not candidates:
            return self._no_trade("No dynamic candidates passed the scanner filters.", ["NO_CANDIDATES"], candidates)

        elapsed = time.monotonic() - self._last_call_monotonic
        if elapsed < self.min_interval_seconds:
            await asyncio.sleep(self.min_interval_seconds - elapsed)

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=self._prompt(candidates, market_context, direction_mode),
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=AITradeDecision,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            self._last_call_monotonic = time.monotonic()
            decision = AITradeDecision.model_validate_json((response.text or "").strip())
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            return self._no_trade(f"AI response validation failed: {exc}", ["AI_PARSE_ERROR"], candidates)
        except Exception as exc:
            return self._no_trade(f"Gemini request failed: {exc}", ["AI_API_ERROR"], candidates)

        allowed = {item["symbol"] for item in candidates}
        flags = list(decision.risk_flags)
        if decision.decision == "TRADE":
            if decision.symbol not in allowed:
                return self._no_trade("Model selected a symbol outside the dynamic candidate list.", flags + ["UNKNOWN_SYMBOL"], candidates)
            if decision.confidence < self.min_confidence:
                return self._no_trade("AI confidence is below the configured threshold.", flags + ["LOW_CONFIDENCE"], candidates)
            if direction_mode == "LONG_ONLY" and decision.direction != "LONG":
                return self._no_trade("SHORT is disabled.", flags + ["SHORT_NOT_ALLOWED"], candidates)
        else:
            decision = decision.model_copy(update={"symbol": "NONE", "direction": "NONE"})

        if not decision.candidates_considered:
            decision = decision.model_copy(update={"candidates_considered": [item["symbol"] for item in candidates]})
        return decision
