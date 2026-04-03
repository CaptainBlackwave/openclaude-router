"""
server.py
---------
Async HTTP server that wraps the smart router, Ollama provider, and Atomic
Chat provider into a single OpenAI-compatible endpoint.

Usage:
    python server.py                     # default port 8080
    PORT=9090 python server.py           # custom port

Endpoints:
    POST /v1/chat/completions   OpenAI-compatible chat completions
    streaming via SSE (stream=true) is supported

    GET /health                 Overall health / readiness
    GET /status                 Smart router statistics (provider scores,
                                 health, model mappings)
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import AsyncIterator, Optional

import httpx
from aiohttp import web

# ---------------------------------------------------------------------------
# Import the existing modules
# ---------------------------------------------------------------------------
from smart_router import SmartRouter
from ollama_provider import ollama_chat, ollama_chat_stream, check_ollama_running
from atomic_chat_provider import (
    atomic_chat,
    atomic_chat_stream,
    check_atomic_chat_running,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
router: Optional[SmartRouter] = None

# ---------------------------------------------------------------------------
# Helpers: Anthropic response -> OpenAI JSON
# ---------------------------------------------------------------------------

def _anthropic_to_openai(anthropic_resp: dict, model: str) -> dict:
    """Convert the Anthropic-format dict returned by ollama/atomic providers
    into an OpenAI chat.completions JSON object."""
    content_blocks = anthropic_resp.get("content", [])
    text = ""
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text", "")

    usage = anthropic_resp.get("usage", {})

    return {
        "id": anthropic_resp.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        },
    }


def _make_openai_error(message: str, status: int = 500) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": "api_error",
                "code": status,
            }
        },
        status=status,
    )


# ---------------------------------------------------------------------------
# Non-streaming chat completion
# ---------------------------------------------------------------------------

async def _call_provider(
    provider_name: str,
    model: str,
    messages: list[dict],
    system: Optional[str],
    max_tokens: int,
    temperature: float,
) -> dict:
    """Call the appropriate provider and return an OpenAI-format dict."""
    if provider_name == "ollama":
        resp = await ollama_chat(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _anthropic_to_openai(resp, model)

    if provider_name == "atomic-chat":
        resp = await atomic_chat(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _anthropic_to_openai(resp, model)

    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()

    if provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        base_url = os.getenv(
            "GEMINI_API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()

    raise ValueError(f"Unknown provider: {provider_name}")


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

async def _anthropic_sse_to_openai_sse(
    anthropic_sse: AsyncIterator, model: str
) -> AsyncIterator[str]:
    """Convert Anthropic-style SSE events from a streaming provider into
    OpenAI-compatible SSE lines."""
    async for line in anthropic_sse:
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if not payload:
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "content_block_delta":
            text = event.get("delta", {}).get("text", "")
            if text:
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": text,
                                "role": "assistant",
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        elif event_type in (
            "message_stop",
            "content_block_stop",
            "message_delta",
        ):
            usage = event.get(
                "usage", event.get("delta", {}).get("usage", {})
            )
            final = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                },
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
            return


async def _stream_provider(
    provider_name: str,
    model: str,
    messages: list[dict],
    system: Optional[str],
    max_tokens: int,
    temperature: float,
    response: web.StreamResponse,
) -> float:
    """Stream from provider and write SSE to the HTTP response.
    Returns the duration in milliseconds."""
    start = time.monotonic()

    try:
        if provider_name == "ollama":
            sse_iter = ollama_chat_stream(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            async for chunk in _anthropic_sse_to_openai_sse(sse_iter, model):
                await response.write(chunk.encode("utf-8"))

        elif provider_name == "atomic-chat":
            sse_iter = atomic_chat_stream(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            async for chunk in _anthropic_sse_to_openai_sse(sse_iter, model):
                await response.write(chunk.encode("utf-8"))

        elif provider_name in ("openai", "gemini"):
            if provider_name == "openai":
                api_key = os.getenv("OPENAI_API_KEY", "")
                base_url = os.getenv(
                    "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
                )
            else:
                api_key = os.getenv("GEMINI_API_KEY", "")
                base_url = os.getenv(
                    "GEMINI_API_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai",
                )
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line:
                            await response.write(
                                line.encode("utf-8") + b"\n"
                            )

        else:
            raise ValueError(
                f"Unknown provider: {provider_name}"
            )

    except Exception as e:
        logger.error(f"Streaming error from {provider_name}: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        await response.write(
            f"data: {json.dumps(error_chunk)}\n\n".encode()
        )

    return (time.monotonic() - start) * 1000


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def handle_chat_completions(
    request: web.Request,
) -> web.Response:
    """POST /v1/chat/completions - OpenAI-compatible chat endpoint."""
    try:
        body = await request.json()
    except Exception:
        return _make_openai_error("Invalid JSON body", 400)

    messages = body.get("messages", [])
    model = body.get("model", "gpt-4.1")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 1.0)
    system = body.get("system", None)

    # Extract system message from messages array if present
    if system is None and messages and messages[0].get("role") == "system":
        system = messages[0].get("content", "")

    try:
        route_result = await router.route(messages, model)
    except Exception as e:
        logger.error(f"Routing error: {e}")
        return _make_openai_error(str(e), 503)

    provider_name = route_result["provider"]
    chosen_model = route_result["model"]

    # --- Non-streaming ---
    if not stream:
        start = time.monotonic()
        try:
            openai_resp = await _call_provider(
                provider_name=provider_name,
                model=chosen_model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            duration_ms = (time.monotonic() - start) * 1000
            await router.record_result(
                provider_name, success=True, duration_ms=duration_ms
            )
            return web.json_response(openai_resp)

        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            await router.record_result(
                provider_name, success=False, duration_ms=duration_ms
            )
            logger.error(
                f"Provider {provider_name} error: {e}"
            )
            return _make_openai_error(
                f"Provider error: {e}", 502
            )

    # --- Streaming (SSE) ---
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    try:
        duration_ms = await _stream_provider(
            provider_name=provider_name,
            model=chosen_model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            response=response,
        )
        await router.record_result(
            provider_name, success=True, duration_ms=duration_ms
        )
    except Exception as e:
        await router.record_result(
            provider_name, success=False, duration_ms=duration_ms
        )
        logger.error(f"Streaming error: {e}")

    await response.write_eof()
    return response


# ---------------------------------------------------------------------------
# Health and status handlers
# ---------------------------------------------------------------------------

async def handle_health(request: web.Request) -> web.Response:
    """GET /health - overall health status."""
    ollama_ok = await check_ollama_running()
    atomic_ok = await check_atomic_chat_running()
    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    gemini_key = bool(os.getenv("GEMINI_API_KEY"))

    status = {
        "status": "ok",
        "router_initialized": (
            router._initialized if router else False
        ),
        "providers": {
            "ollama": "ok" if ollama_ok else "unreachable",
            "atomic-chat": "ok" if atomic_ok else "unreachable",
            "openai": "configured" if openai_key else "no-api-key",
            "gemini": "configured" if gemini_key else "no-api-key",
        },
    }
    return web.json_response(status)


async def handle_status(request: web.Request) -> web.Response:
    """GET /status - detailed smart router statistics."""
    provider_status = router.status() if router else []
    provider_catalog = {
        "ollama": {
            "base_url": os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
        },
        "atomic-chat": {
            "base_url": os.getenv(
                "ATOMIC_CHAT_BASE_URL", "http://127.0.0.1:1337"
            ),
        },
        "openai": {
            "base_url": os.getenv(
                "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
            ),
        },
        "gemini": {
            "base_url": os.getenv(
                "GEMINI_API_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            ),
        },
    }

    return web.json_response(
        {
            "strategy": router.strategy,
            "fallback_enabled": router.fallback_enabled,
            "providers": provider_status,
            "model_mappings": {
                "ollama": {
                    "big": os.getenv("BIG_MODEL", "llama3:8b"),
                    "small": os.getenv("SMALL_MODEL", "llama3:8b"),
                },
                "atomic-chat": {
                    "big": os.getenv("BIG_MODEL", "llama3:8b"),
                    "small": os.getenv("SMALL_MODEL", "llama3:8b"),
                },
                "openai": {
                    "big": os.getenv("BIG_MODEL", "gpt-4.1"),
                    "small": os.getenv("SMALL_MODEL", "gpt-4.1-mini"),
                },
            },
            "provider_catalog": provider_catalog,
        }
    )


# ---------------------------------------------------------------------------
# Application factory and main
# ---------------------------------------------------------------------------


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/status", handle_status)
    return app


async def _init_router() -> None:
    global router
    router = SmartRouter()
    await router.initialize()


def main() -> None:
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))

    logger.info(
        f"Starting openclaude smart router server on {host}:{port}"
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_init_router())

    app = create_app()
    web.run_app(app, host=host, port=port, handle_signals=True)


if __name__ == "__main__":
    main()
