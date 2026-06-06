"""
HF Inference Providers インフラ層 — データ合成用

asyncio ベースの高スループット実装。AsyncInferenceClient でコネクションを多重化し、
Semaphore でプロバイダーのレート制限に対応する。

同期 API (chat_complete / batch_chat_complete) はそのまま使える。
大量バッチ処理には非同期 API (async_chat_complete / async_batch_chat_complete) を推奨。

使用例（同期）:
    config = InferenceConfig(model="meta-llama/Llama-3.1-8B-Instruct", provider="auto")
    client = HFInferenceClient(config)
    result = client.chat_complete([{"role": "user", "content": "Hello"}])
    print(result.content)

使用例（非同期・高スループット）:
    import asyncio
    results = asyncio.run(client.async_batch_chat_complete(batches, max_concurrency=32))
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal, Optional

from huggingface_hub import AsyncInferenceClient, InferenceClient

logger = logging.getLogger(__name__)

Provider = Literal[
    "auto",
    "hf-inference",
    "together",
    "fireworks-ai",
    "groq",
    "cerebras",
    "deepinfra",
    "nebius",
    "novita",
    "nscale",
    "cohere",
    "sambanova",
    "hyperbolic",
    "fal-ai",
    "featherless-ai",
    "openai",
    "replicate",
    "scaleway",
    "publicai",
    "zai-org",
    "black-forest-labs",
]

# OpenAI形式のメッセージ辞書
Message = dict[str, str]


@dataclass
class InferenceConfig:
    """HF Inference Provider の呼び出し設定。"""

    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    provider: Provider = "auto"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    seed: Optional[int] = None
    stop: list[str] = field(default_factory=list)
    # リトライ設定
    max_retries: int = 3
    retry_delay: float = 1.0  # 初回待機秒数（指数バックオフ）
    # クライアント設定
    timeout: float = 60.0


@dataclass
class CompletionResult:
    """単一チャット補完の結果。"""

    content: str
    model: str
    provider: str
    usage: dict[str, int]
    latency_s: float


class HFInferenceClient:
    """
    HF Inference Providers ラッパー。

    同期 API:
        chat_complete()         — 単一リクエスト、指数バックオフ付きリトライ
        batch_chat_complete()   — ThreadPoolExecutor による並列実行（後方互換）

    非同期 API（高スループット推奨）:
        async_chat_complete()       — 単一リクエストの非同期版
        async_batch_chat_complete() — asyncio + Semaphore で多重化。
                                      スレッドオーバーヘッドなし、max_concurrency=32+ が現実的。
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        token: str | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self._token = token or os.environ.get("HF_TOKEN")
        if not self._token:
            raise ValueError(
                "HF_TOKEN 環境変数またはコンストラクタの token 引数が必要です"
            )
        self._sync_client = self._build_sync_client(self.config)
        # AsyncInferenceClient はセッションを内部で管理するのでインスタンスを再利用する
        self._async_client = self._build_async_client(self.config)

    # ------------------------------------------------------------------
    # 同期 API（後方互換）
    # ------------------------------------------------------------------

    def chat_complete(
        self,
        messages: list[Message],
        *,
        config_override: InferenceConfig | None = None,
    ) -> CompletionResult:
        """
        単一チャット補完。指数バックオフ付きリトライ。

        Raises:
            RuntimeError: max_retries 回失敗した場合。
        """
        cfg = config_override or self.config
        client = self._resolve_sync_client(cfg)

        last_exc: Exception | None = None
        delay = cfg.retry_delay

        for attempt in range(cfg.max_retries):
            try:
                t0 = time.monotonic()
                response = client.chat_completion(
                    messages=messages,
                    model=cfg.model,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    seed=cfg.seed,
                    stop=cfg.stop or None,
                )
                latency = time.monotonic() - t0
                return _parse_response(response, cfg, latency)

            except Exception as exc:
                last_exc = exc
                if attempt < cfg.max_retries - 1:
                    logger.warning(
                        "試行 %d/%d 失敗: %s。%.1f秒後にリトライします...",
                        attempt + 1, cfg.max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2

        raise RuntimeError(f"{cfg.max_retries} 回の試行がすべて失敗しました") from last_exc

    def batch_chat_complete(
        self,
        message_batches: list[list[Message]],
        *,
        max_workers: int = 8,
        config_override: InferenceConfig | None = None,
    ) -> list[CompletionResult | Exception]:
        """
        スレッドプールによる並列補完（後方互換）。

        大量バッチには async_batch_chat_complete() を推奨。
        失敗アイテムは Exception として返す（raise しない）。
        """
        results: list[CompletionResult | Exception] = [None] * len(message_batches)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.chat_complete, msgs, config_override=config_override): i
                for i, msgs in enumerate(message_batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("バッチアイテム %d 失敗: %s", idx, exc)
                    results[idx] = exc
        return results

    # ------------------------------------------------------------------
    # 非同期 API（高スループット）
    # ------------------------------------------------------------------

    async def async_chat_complete(
        self,
        messages: list[Message],
        *,
        config_override: InferenceConfig | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> CompletionResult:
        """
        単一チャット補完の非同期版。指数バックオフ付きリトライ。

        semaphore を渡すと同時実行数を制限できる（batch 内部から呼ばれる場合は自動付与）。
        """
        cfg = config_override or self.config
        client = self._resolve_async_client(cfg)

        async def _do() -> CompletionResult:
            last_exc: Exception | None = None
            delay = cfg.retry_delay
            for attempt in range(cfg.max_retries):
                try:
                    t0 = time.monotonic()
                    response = await client.chat_completion(
                        messages=messages,
                        model=cfg.model,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        seed=cfg.seed,
                        stop=cfg.stop or None,
                    )
                    latency = time.monotonic() - t0
                    return _parse_response(response, cfg, latency)
                except Exception as exc:
                    last_exc = exc
                    if attempt < cfg.max_retries - 1:
                        logger.warning(
                            "試行 %d/%d 失敗: %s。%.1f秒後にリトライします...",
                            attempt + 1, cfg.max_retries, exc, delay,
                        )
                        await asyncio.sleep(delay)
                        delay *= 2
            raise RuntimeError(f"{cfg.max_retries} 回の試行がすべて失敗しました") from last_exc

        if semaphore is not None:
            async with semaphore:
                return await _do()
        return await _do()

    async def async_batch_chat_complete(
        self,
        message_batches: list[list[Message]],
        *,
        max_concurrency: int = 32,
        config_override: InferenceConfig | None = None,
    ) -> list[CompletionResult | Exception]:
        """
        asyncio + Semaphore による高スループットバッチ補完。

        スレッドを使わないため、max_concurrency=32 以上でもオーバーヘッドが小さい。
        失敗アイテムは Exception として返す（raise しない）。

        Args:
            message_batches: メッセージリストのリスト。
            max_concurrency: 同時実行数の上限（Semaphore）。
            config_override: 全リクエスト共通の設定上書き。
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _safe(i: int, msgs: list[Message]) -> tuple[int, CompletionResult | Exception]:
            try:
                result = await self.async_chat_complete(
                    msgs,
                    config_override=config_override,
                    semaphore=semaphore,
                )
                return i, result
            except Exception as exc:
                logger.error("バッチアイテム %d 失敗: %s", i, exc)
                return i, exc

        tasks = [_safe(i, msgs) for i, msgs in enumerate(message_batches)]
        pairs = await asyncio.gather(*tasks)

        results: list[CompletionResult | Exception] = [None] * len(message_batches)  # type: ignore[list-item]
        for idx, result in pairs:
            results[idx] = result
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_sync_client(self, cfg: InferenceConfig) -> InferenceClient:
        return InferenceClient(
            model=cfg.model,
            provider=cfg.provider,
            token=self._token,
            timeout=cfg.timeout,
        )

    def _build_async_client(self, cfg: InferenceConfig) -> AsyncInferenceClient:
        return AsyncInferenceClient(
            model=cfg.model,
            provider=cfg.provider,
            token=self._token,
            timeout=cfg.timeout,
        )

    def _resolve_sync_client(self, cfg: InferenceConfig) -> InferenceClient:
        if (
            cfg.provider == self.config.provider
            and cfg.model == self.config.model
            and cfg.timeout == self.config.timeout
        ):
            return self._sync_client
        return self._build_sync_client(cfg)

    def _resolve_async_client(self, cfg: InferenceConfig) -> AsyncInferenceClient:
        if (
            cfg.provider == self.config.provider
            and cfg.model == self.config.model
            and cfg.timeout == self.config.timeout
        ):
            return self._async_client
        return self._build_async_client(cfg)


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------


def synthesize_from_template(
    client: HFInferenceClient,
    system_prompt: str,
    user_prompts: list[str],
    *,
    max_concurrency: int = 32,
) -> list[str | Exception]:
    """
    システムプロンプト + ユーザープロンプトのリストからアシスタント回答を一括生成する。

    asyncio ベースの高スループット実装。
    失敗アイテムは Exception として返す（raise しない）。
    """
    batches: list[list[Message]] = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p},
        ]
        for p in user_prompts
    ]
    results = asyncio.run(
        client.async_batch_chat_complete(batches, max_concurrency=max_concurrency)
    )
    return [r.content if isinstance(r, CompletionResult) else r for r in results]


async def async_synthesize_from_template(
    client: HFInferenceClient,
    system_prompt: str,
    user_prompts: list[str],
    *,
    max_concurrency: int = 32,
) -> list[str | Exception]:
    """synthesize_from_template の非同期版。既存の event loop 内から呼ぶ場合に使う。"""
    batches: list[list[Message]] = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p},
        ]
        for p in user_prompts
    ]
    results = await client.async_batch_chat_complete(
        batches, max_concurrency=max_concurrency
    )
    return [r.content if isinstance(r, CompletionResult) else r for r in results]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_response(response, cfg: InferenceConfig, latency: float) -> CompletionResult:
    choice = response.choices[0]
    usage: dict[str, int] = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
        }
    return CompletionResult(
        content=choice.message.content or "",
        model=response.model or cfg.model,
        provider=cfg.provider,
        usage=usage,
        latency_s=latency,
    )
