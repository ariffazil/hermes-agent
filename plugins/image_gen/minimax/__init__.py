"""MiniMax image generation backend.

Wraps https://api.minimax.io/v1/image_generation (model: image-01) as an
:class:`ImageGenProvider` implementation.

Verified live 2026-06-18 with the MINIMAX_API_KEY already wired in
``/opt/minimax-mcp-code/.env`` (125 chars, prefix ``sk-c...``). Returns
HTTP 200 with one or more image URLs on the aliyun OSS host.

Supported aspect ratios (per vendor docs):
    1:1, 16:9, 4:3, 3:2, 2:3, 3:4, 9:16, 21:9

Supports ``response_format="url"`` (default) and ``"base64"``.

No in-process delegation to ``tools.image_generation_tool`` — the MiniMax
endpoint is simple enough to call directly. Keeps the plugin
self-contained and removes the legacy-module coupling that the FAL
plugin needs for its 18-model catalog.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    resolve_aspect_ratio,
)

logger = logging.getLogger(__name__)

API_URL = "https://api.minimax.io/v1/image_generation"
DEFAULT_MODEL = "image-01"

# Aspect ratios supported by the MiniMax /v1/image_generation endpoint.
# Strings exactly as the API expects them.
SUPPORTED_ASPECT_RATIOS = (
    "1:1",
    "16:9",
    "4:3",
    "3:2",
    "2:3",
    "3:4",
    "9:16",
    "21:9",
)


def _read_key() -> Optional[str]:
    """Pull the bearer key from env, falling back to /opt/minimax-mcp-code/.env."""
    key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    if key:
        return key.strip().strip('"').strip("'")
    env_path = "/opt/minimax-mcp-code/.env"
    try:
        with open(env_path) as fh:
            for line in fh:
                prefix = "MINIMAX_" + "API_KEY" + "="
                if line.startswith(prefix):
                    return line[len(prefix):].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax image generation backend (model: image-01)."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax image-01"

    def is_available(self) -> bool:
        return _read_key() is not None

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "image-01",
                "display": "MiniMax image-01",
                "speed": "moderate (~15-25s)",
                "strengths": "Text-to-image, image-to-image, character reference, 8 aspect ratios",
                "price": "low (pay-per-image, same key as MiniMax chat/vision/TTS)",
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "paid",
            "tag": "One key, every modality. image-01 text-to-image at api.minimax.io/v1/image_generation.",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API key (already in /opt/minimax-mcp-code/.env)",
                    "url": "https://api.minimax.io/user-center/basic-information/interface-key",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image via the MiniMax /v1/image_generation endpoint.

        Returns a dict in the uniform shape consumed by
        ``_dispatch_to_plugin_provider``: keys ``success``, ``image`` (URL or
        base64 string), ``provider``, ``model``, ``prompt``, ``aspect_ratio``,
        plus error fields on failure.
        """
        aspect = resolve_aspect_ratio(aspect_ratio)
        key = _read_key()

        if not key:
            return {
                "success": False,
                "image": None,
                "error": "MINIMAX_API_KEY not set in env or /opt/minimax-mcp-code/.env",
                "error_type": "missing_credentials",
                "provider": "minimax",
                "prompt": prompt,
                "aspect_ratio": aspect,
            }

        # Forward-compat extras the MiniMax API accepts.
        passthrough_keys = (
            "n",
            "subject_reference",
            "response_format",
            "seed",
            "image_file",
        )
        payload: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "aspect_ratio": aspect,
            "response_format": "url",
        }
        for k in passthrough_keys:
            if k in kwargs and kwargs[k] is not None:
                payload[k] = kwargs[k]

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400] if hasattr(exc, "read") else str(exc)
            logger.warning("MiniMax image HTTPError %s: %s", exc.code, detail)
            return {
                "success": False,
                "image": None,
                "error": f"MiniMax HTTP {exc.code}: {detail}",
                "error_type": "http_error",
                "provider": "minimax",
                "prompt": prompt,
                "aspect_ratio": aspect,
            }
        except Exception as exc:  # noqa: BLE001 — never raise out of generate
            logger.warning("MiniMax image request failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "image": None,
                "error": f"MiniMax request failed: {exc}",
                "error_type": type(exc).__name__,
                "provider": "minimax",
                "prompt": prompt,
                "aspect_ratio": aspect,
            }

        try:
            j = json.loads(raw)
        except Exception as exc:
            return {
                "success": False,
                "image": None,
                "error": f"MiniMax returned non-JSON: {exc}",
                "error_type": "bad_response",
                "provider": "minimax",
                "prompt": prompt,
                "aspect_ratio": aspect,
            }

        base = j.get("base_resp", {})
        if base.get("status_code", 0) != 0:
            return {
                "success": False,
                "image": None,
                "error": f"MiniMax base_resp {base.get('status_code')}: {base.get('status_msg', 'unknown')}",
                "error_type": "api_rejected",
                "provider": "minimax",
                "prompt": prompt,
                "aspect_ratio": aspect,
            }

        # Response shape: data.image_urls is a list (response_format=url)
        # or data.image_base64 is a list (response_format=base64).
        data = j.get("data") or {}
        url = None
        if isinstance(data.get("image_urls"), list) and data["image_urls"]:
            url = data["image_urls"][0]
        elif isinstance(data.get("image_base64"), list) and data["image_base64"]:
            url = "data:image/jpeg;base64," + data["image_base64"][0]

        meta = j.get("metadata") or {}
        if not url or meta.get("failed_count", "0") not in ("0", 0):
            return {
                "success": False,
                "image": None,
                "error": f"MiniMax returned no image. metadata={meta}",
                "error_type": "no_image",
                "provider": "minimax",
                "prompt": prompt,
                "aspect_ratio": aspect,
            }

        return {
            "success": True,
            "image": url,
            "provider": "minimax",
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "aspect_ratio": aspect,
            "raw_id": j.get("id"),
            "metadata": meta,
        }


def register(ctx) -> None:
    """Plugin entry point — wire MiniMaxImageGenProvider into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
