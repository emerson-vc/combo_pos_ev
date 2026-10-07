"""Async Kalshi REST client: RSA-PSS request signing + simple per-second rate limiting."""

import asyncio
import base64
import time
from urllib.parse import urlparse

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from app import config


class RateLimiter:
    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


class KalshiError(Exception):
    def __init__(self, status: int, body: str, path: str):
        super().__init__(f"Kalshi {status} on {path}: {body[:300]}")
        self.status = status


class KalshiClient:
    def __init__(
        self,
        base_url: str = config.KALSHI_BASE_URL,
        key_id: str | None = config.KALSHI_KEY_ID,
        private_key_path=config.KALSHI_PRIVATE_KEY_PATH,
        reads_per_sec: float = 10,
        writes_per_sec: float = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.path_prefix = urlparse(self.base_url).path  # "/trade-api/v2", part of the signed path
        self.key_id = key_id
        self._key = None
        if key_id and private_key_path and private_key_path.exists():
            self._key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
        self._http = httpx.AsyncClient(timeout=20)
        self._reads = RateLimiter(reads_per_sec)
        self._writes = RateLimiter(writes_per_sec)

    @property
    def has_auth(self) -> bool:
        return self._key is not None

    def _sign_headers(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        msg = (ts + method + self.path_prefix + path.split("?")[0]).encode()
        sig = self._key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    async def request(self, method: str, path: str, *, params=None, json=None, auth=False):
        await (self._reads if method == "GET" else self._writes).wait()
        headers = {}
        if auth or self.has_auth:
            if not self.has_auth:
                raise RuntimeError(f"{method} {path} needs a Kalshi API key (see .env.example)")
            headers = self._sign_headers(method, path)
        for attempt in range(4):
            try:
                resp = await self._http.request(method, self.base_url + path, params=params, json=json,
                                                headers=headers)
            except httpx.TransportError:  # dropped connection / timeout: back off and retry
                if attempt == 3 or method == "POST":  # a retried POST could create a duplicate RFQ
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))
                if self.has_auth:
                    headers = self._sign_headers(method, path)
                continue
            if resp.status_code == 429:
                await asyncio.sleep(1.0 * (attempt + 1))
                if self.has_auth:
                    headers = self._sign_headers(method, path)
                continue
            break
        if resp.status_code >= 400:
            raise KalshiError(resp.status_code, resp.text, path)
        return resp.json() if resp.content else {}

    async def get(self, path, params=None, auth=False):
        return await self.request("GET", path, params=params, auth=auth)

    async def post(self, path, json):
        return await self.request("POST", path, json=json, auth=True)

    async def delete(self, path):
        return await self.request("DELETE", path, auth=True)

    async def paginate(self, path, key, params=None):
        params = dict(params or {})
        out = []
        while True:
            data = await self.get(path, params)
            out.extend(data.get(key, []))
            cursor = data.get("cursor")
            if not cursor:
                return out
            params["cursor"] = cursor

    async def close(self):
        await self._http.aclose()
