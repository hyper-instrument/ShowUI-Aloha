"""Smoke-test a third-party Anthropic-compatible vendor (e.g. gpugeek.com).

Sends a minimal Computer-Use request (1024x768 blank PNG + a single instruction)
to each configured model and reports availability, basic shape of the response,
and round-trip latency.

Required env vars (any auth method works):
  ANTHROPIC_BASE_URL   e.g. https://api.gpugeek.com
  ANTHROPIC_AUTH_TOKEN bearer token (preferred for vendors using Authorization: Bearer)
  # or ANTHROPIC_API_KEY for x-api-key style

Optional:
  ANTHROPIC_MODELS     comma-separated override; otherwise the script tests
                       the six gpugeek "Vendor2/Claude-*" models.
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import time
from dataclasses import dataclass

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    print("anthropic SDK is not installed; run `pip install -r requirements.txt`.")
    sys.exit(2)

try:
    from PIL import Image
except ImportError:
    print("Pillow is required; run `pip install Pillow`.")
    sys.exit(2)


DEFAULT_MODELS = [
    "Vendor2/Claude-4.5-Sonnet",
    "Vendor2/Claude-4.6-Sonnet",
    "Vendor2/Claude-4.7-Sonnet",
    "Vendor2/Claude-4.5-Opus",
    "Vendor2/Claude-4.6-Opus",
    "Vendor2/Claude-4.7-Opus",
]


_VERSION_RE = re.compile(r"(\d+)[.\-_](\d+)")


def resolve_tool_version(model: str) -> tuple[str, str]:
    """Return (computer_tool_type, beta_flag) for the given model name.

    Per Anthropic docs:
      - computer-use-2025-11-24: Opus 4.7, Opus 4.6, Sonnet 4.6, Opus 4.5
      - computer-use-2025-01-24: Sonnet 4.5, Haiku 4.5, Opus 4.1, Sonnet 4, Opus 4, Sonnet 3.7

    Robust to both Anthropic-native names ("claude-opus-4-7-...") and
    vendor names ("Vendor2/Claude-4.7-Opus") where the family token may
    appear before or after the version number.
    """
    n = model.lower()
    family = "opus" if "opus" in n else "sonnet" if "sonnet" in n else "haiku" if "haiku" in n else ""
    m = _VERSION_RE.search(n)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2)) if m else 0

    needs_v2 = False
    if family == "opus" and (major, minor) >= (4, 5):
        needs_v2 = True
    elif family == "sonnet" and (major, minor) >= (4, 6):
        needs_v2 = True

    if needs_v2:
        return "computer_20251124", "computer-use-2025-11-24"
    return "computer_20250124", "computer-use-2025-01-24"


def make_blank_screenshot_b64(width: int = 1024, height: int = 768) -> str:
    img = Image.new("RGB", (width, height), (245, 245, 245))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class Result:
    model: str
    ok: bool
    elapsed_s: float
    stop_reason: str | None
    text_blocks: int
    tool_blocks: int
    input_tokens: int | None
    output_tokens: int | None
    error: str | None


def build_client() -> anthropic.Anthropic:
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")

    if not (auth_token or api_key):
        print(
            "ERROR: please export ANTHROPIC_AUTH_TOKEN (preferred for vendors) "
            "or ANTHROPIC_API_KEY before running this script.",
            file=sys.stderr,
        )
        sys.exit(2)

    kwargs: dict = {}
    if base_url:
        kwargs["base_url"] = base_url
    if auth_token:
        kwargs["auth_token"] = auth_token
    elif api_key:
        kwargs["api_key"] = api_key
    return anthropic.Anthropic(**kwargs)


def probe_model(client: anthropic.Anthropic, model: str, screenshot_b64: str) -> Result:
    tool_type, beta = resolve_tool_version(model)
    print(f"\n=== {model} ===  tool={tool_type}  beta={beta}", flush=True)
    t0 = time.time()
    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=256,
            tools=[
                {
                    "type": tool_type,
                    "name": "computer",
                    "display_width_px": 1024,
                    "display_height_px": 768,
                    "display_number": 1,
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Use the `computer` tool to perform a "
                                "left_click at coordinate [512, 384] now. "
                                "Do not respond with text first; call the tool."
                            ),
                        },
                    ],
                }
            ],
            betas=[beta],
        )
        elapsed = time.time() - t0
        text_n = sum(1 for b in resp.content if getattr(b, "type", "") == "text")
        tool_n = sum(1 for b in resp.content if getattr(b, "type", "") == "tool_use")
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) if usage else None
        out_tok = getattr(usage, "output_tokens", None) if usage else None
        stop = getattr(resp, "stop_reason", None)
        print(
            f"  OK  stop={stop}  text={text_n}  tool_use={tool_n}  "
            f"tokens(in/out)={in_tok}/{out_tok}  time={elapsed:.2f}s",
            flush=True,
        )
        return Result(
            model=model,
            ok=True,
            elapsed_s=elapsed,
            stop_reason=stop,
            text_blocks=text_n,
            tool_blocks=tool_n,
            input_tokens=in_tok,
            output_tokens=out_tok,
            error=None,
        )
    except Exception as e:
        elapsed = time.time() - t0
        msg = f"{type(e).__name__}: {e}"
        print(f"  ERR  time={elapsed:.2f}s  {msg}", flush=True)
        return Result(
            model=model,
            ok=False,
            elapsed_s=elapsed,
            stop_reason=None,
            text_blocks=0,
            tool_blocks=0,
            input_tokens=None,
            output_tokens=None,
            error=msg,
        )


def main() -> int:
    models_env = os.getenv("ANTHROPIC_MODELS")
    if models_env:
        models = [m.strip() for m in models_env.split(",") if m.strip()]
    else:
        models = DEFAULT_MODELS

    base_url = os.getenv("ANTHROPIC_BASE_URL", "<unset, will hit api.anthropic.com>")
    print(f"Vendor base_url: {base_url}")
    print(f"Models to test : {models}")

    client = build_client()
    screenshot_b64 = make_blank_screenshot_b64()

    results = [probe_model(client, m, screenshot_b64) for m in models]

    print("\n=== Summary ===")
    print(f"{'STATUS':6s}  {'TIME':>7s}  {'IN/OUT TOKENS':>14s}  {'TOOL':>4s}  MODEL")
    for r in results:
        status = "OK" if r.ok else "ERR"
        tok = f"{r.input_tokens or '-'}/{r.output_tokens or '-'}"
        print(
            f"{status:6s}  {r.elapsed_s:6.2f}s  {tok:>14s}  {r.tool_blocks:>4d}  {r.model}"
            + ("" if r.ok else f"   :: {r.error}")
        )

    failures = [r for r in results if not r.ok]
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
