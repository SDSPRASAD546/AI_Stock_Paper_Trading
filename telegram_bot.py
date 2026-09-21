from __future__ import annotations

import asyncio
import os
import requests

from dotenv import load_dotenv
from telebot.async_telebot import AsyncTeleBot

load_dotenv()


class TelegramBot:
    """Async Telegram client for Upstox OAuth and paper-trading alerts."""

    def __init__(self):
        self.token = os.getenv("BOT_TOKEN", "").strip()
        chat_id = os.getenv("CHAT_ID", "").strip()
        if not self.token:
            raise ValueError("BOT_TOKEN is not set")
        if not chat_id:
            raise ValueError("CHAT_ID is not set")

        self.chat_id = int(chat_id)
        self.bot = AsyncTeleBot(self.token)
        self.reply_event = asyncio.Event()
        self.received_code: str | None = None

        @self.bot.message_handler(func=lambda message: message.chat.id == self.chat_id)
        async def handle_message(message):
            text = (message.text or "").strip()
            if not text:
                return
            self.received_code = text
            masked = text[:4] + "..." + text[-4:] if len(text) > 8 else "<redacted>"
            print(f"Telegram auth response received: {masked}")
            self.reply_event.set()

    async def _send_with_retry(self, text: str, attempts: int = 3) -> bool:
        for attempt in range(1, attempts + 1):
            try:
                await asyncio.wait_for(
                    self.bot.send_message(self.chat_id, text),
                    timeout=20,
                )
                return True
            except Exception as exc:
                print(f"Telegram send attempt {attempt}/{attempts} failed: {exc}")
                if attempt < attempts:
                    await asyncio.sleep(2 * attempt)
        return False

    async def send_link_and_wait_for_code(self, link: str, timeout: int = 300) -> str | None:
        self.received_code = None
        self.reply_event.clear()
        if not await self._send_with_retry(link):
            return None
        print("Upstox login link sent. Waiting for the full redirect URL/code in Telegram...")
        try:
            await asyncio.wait_for(self.reply_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print("Upstox authentication timeout.")
            await self._send_with_retry(
                f"Authentication timed out after {max(1, int(timeout // 60))} minutes. Run the bot again to retry."
            )
            return None
        return self.received_code

    async def send_message(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"

        payload = {
            "chat_id": self.chat_id,
            "text": text,
        }

        def send():
            return requests.post(
                url,
                json=payload,
                timeout=10,
            )

        for attempt in range(1, 4):
            try:
                response = await asyncio.to_thread(send)

                if response.status_code == 200:
                    return True

                print(
                    f"Telegram send attempt {attempt}/3 failed: "
                    f"HTTP {response.status_code} - {response.text}"
                )

            except requests.RequestException as exc:
                print(
                    f"Telegram send attempt {attempt}/3 failed: {exc}"
                )

            if attempt < 3:
                await asyncio.sleep(2)

        return False

    async def start(self):
        print("Telegram polling started...")
        await self.bot.polling(non_stop=True)

    async def close(self):
        try:
            await self.bot.close_session()
        except Exception:
            pass
