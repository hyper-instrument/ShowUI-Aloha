"""Claude actor that emits Aloha actions via a *custom* tool use call.

Why this exists
---------------
Anthropic's "computer use" beta ships its own predefined tool schema
(`{action: "left_click", coordinate: [..]}`). Some vendors (e.g. gpugeek)
emulate that beta with prompt scaffolding rather than implementing it for
real, so the `tool_use.input` they emit drifts wildly across calls.

This agent sidesteps the whole computer-use beta. It registers a *custom*
tool, ``aloha_action``, whose ``input_schema`` is the Aloha action format
itself, and forces the model to call exactly that tool. Because Anthropic
(and any compatible vendor) validates tool inputs against the supplied
``input_schema`` server-side, the model can no longer return free-form
text, markdown, or arbitrarily named keys: the keys come straight from our
schema.

Layering vs. ``ClaudeComputerUseAgent``
---------------------------------------
``ClaudeComputerUseAgent`` (kept untouched) -> Anthropic Computer Use beta
``ClaudeAlohaComputerUseAgent`` (this file) -> standard tool use, custom schema

We deliberately keep the two agents as parallel files for readability; if a
third Claude variant ever shows up we should refactor to a shared base.
"""

from __future__ import annotations

import os

from jinja2 import Environment, FileSystemLoader

from ui_aloha.act.gui_agent.llm.llm_utils import encode_image
from ui_aloha.act.utils.path_utils import prompt_templates_path

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Custom tool schema
# ---------------------------------------------------------------------------
#
# Each property maps 1:1 to a field that AlohaExecutor's parsers already know
# how to consume. The model picks an `action` from the enum; the schema makes
# the rest of the fields optional but constrains *types*. Per-action field
# requirements are described in the system prompt rather than encoded as
# JSON Schema oneOf/if-then because (a) it keeps the schema simple, (b)
# Anthropic's tool input validation handles types/enum but not cross-field
# conditionals, and (c) the executor itself ignores fields that don't apply.
#
ALOHA_TOOL = {
    "name": "aloha_action",
    "description": (
        "Emit exactly one low-level GUI action to advance the user's task. "
        "Prefer normalized coordinates in [0,1] x [0,1] for CLICK/MOVE (fraction "
        "of screenshot width/height); integers in the documented 1024×768 reference "
        "space are also accepted."
    ),
    "input_schema": {
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": {
                "type": "string",
                "description": "Which low-level GUI action to perform.",
                "enum": [
                    "CLICK",
                    "RIGHT_CLICK",
                    "DOUBLE_CLICK",
                    "TRIPLE_CLICK",
                    "MOVE",
                    "INPUT",
                    "KEY",
                    "HOTKEY",
                    "ENTER",
                    "ESC",
                    "DRAG",
                    "SCROLL",
                    "WAIT",
                    "STOP",
                ],
            },
            "position": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "[x, y] target: normalized fractions in [0,1] (recommended), "
                    "or pixel coords in 1024×768 reference space. Required for "
                    "CLICK / RIGHT_CLICK / DOUBLE_CLICK / TRIPLE_CLICK / MOVE / "
                    "SCROLL / DRAG end-point. Use [0, 0] when not applicable."
                ),
            },
            "text": {
                "type": "string",
                "description": "INPUT: the literal text to type.",
            },
            "key": {
                "type": "string",
                "description": (
                    "KEY: a single named key, e.g. 'Return', 'Tab', 'Escape'."
                ),
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "HOTKEY: chord, e.g. ['cmd', 's'] or "
                    "['ctrl', 'shift', 't']."
                ),
            },
            "drag_from": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "DRAG: starting [x, y]. `position` is the DRAG end-point."
                ),
            },
            "scroll_amount": {
                "type": "integer",
                "description": (
                    "SCROLL: positive = down / right, negative = up / left."
                ),
            },
            "wait_seconds": {
                "type": "number",
                "description": "WAIT: seconds to sleep.",
            },
            "stop_summary": {
                "type": "string",
                "description": (
                    "STOP: one-line summary of why the task is complete."
                ),
            },
        },
        "additionalProperties": False,
    },
}


class ClaudeAlohaComputerUseAgent:
    """Claude actor that emits Aloha action JSON via a custom tool use call."""

    def __init__(
        self,
        api_key: str | None = None,
        logger=None,
        base_url: str | None = None,
        auth_token: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ):
        # Same credential resolution as the tool-use beta variant: prefer
        # Bearer (vendor) when available, fall back to x-api-key (official).
        base_url = base_url or os.getenv("ANTHROPIC_BASE_URL") or None
        auth_token = auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN") or None
        api_key = (
            api_key
            or os.getenv("CLAUDE_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or None
        )

        client_kwargs: dict = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if auth_token:
            client_kwargs["auth_token"] = auth_token
        elif api_key:
            client_kwargs["api_key"] = api_key

        if ANTHROPIC_AVAILABLE and (auth_token or api_key):
            self.client = anthropic.Anthropic(**client_kwargs)
        else:
            self.client = None

        self.logger = logger
        self.base_url = base_url
        self.model = model or os.getenv("ANTHROPIC_MODEL") or _DEFAULT_MODEL
        self.max_tokens = max_tokens

        # Display the agent advertises in the prompt. Coordinates Claude emits
        # are in this space; we rescale to the executor's 1920x1080 frame.
        self.DISPLAY_WIDTH = 1024
        self.DISPLAY_HEIGHT = 768
        self.TARGET_WIDTH = 1920
        self.TARGET_HEIGHT = 1080
        self._executor_frame_w = self.TARGET_WIDTH
        self._executor_frame_h = self.TARGET_HEIGHT

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def execute(self, instruction, screenshot_path, system_prompt, logging_dir):
        """Send one user turn and return ``(action_json, complete_flag)``."""
        if not ANTHROPIC_AVAILABLE or not self.client:
            error_msg = "Anthropic library not available or no credentials provided"
            if self.logger:
                self.logger.logger.error(error_msg)
            return {"action": "ERROR", "value": error_msg, "position": [0, 0]}, False

        screenshot_base64 = encode_image(screenshot_path)

        try:
            try:
                from PIL import Image

                with Image.open(screenshot_path) as im:
                    self._executor_frame_w, self._executor_frame_h = im.size
            except Exception:
                self._executor_frame_w = self.TARGET_WIDTH
                self._executor_frame_h = self.TARGET_HEIGHT

            templates_dir = prompt_templates_path()
            env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            user_text = env.get_template("actor/user_cua.txt").render(task=instruction)

            if self.logger:
                self.logger.logger.info(
                    f"claude_aloha_computer_use: model={self.model} "
                    f"base_url={self.base_url or 'default'} "
                    "(custom tool use, schema-enforced)"
                )

            # Standard tool use call. No `betas=[...]`. We register exactly
            # one custom tool (`aloha_action`) and force the model to call it.
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=[ALOHA_TOOL],
                tool_choice={"type": "tool", "name": "aloha_action"},
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                }],
            )

            if self.logger:
                self.logger.log_json(
                    {"response": str(response)},
                    "actor_claude_aloha_computer_use_raw_response.json",
                    logging_dir,
                )

            action_json = self._parse_response(response)

            if self.logger:
                self.logger.log_json(
                    action_json,
                    "actor_claude_aloha_computer_use_parsed_action.json",
                    logging_dir,
                )

            return action_json, action_json.get("action") == "STOP"

        except Exception as e:
            error_msg = f"Error processing claude-aloha-computer-use response: {e}"
            if self.logger:
                self.logger.logger.error(error_msg)
                self.logger.log_error(
                    e, {"mode": "claude-aloha-computer-use"}, target_dir=logging_dir
                )
            return {"action": "ERROR", "value": str(e), "position": [0, 0]}, False

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _scale_xy(self, coord) -> list[int]:
        """Map tool coordinates into executor pixel space (current screenshot size).

        Models often return normalized [0, 1] fractions (vendor drift). Our schema
        also documents 1024×768 reference pixels; those are scaled proportionally.
        """
        fw = getattr(self, "_executor_frame_w", self.TARGET_WIDTH)
        fh = getattr(self, "_executor_frame_h", self.TARGET_HEIGHT)
        if not coord or not isinstance(coord, (list, tuple)) or len(coord) < 2:
            return [0, 0]
        try:
            x = float(coord[0])
            y = float(coord[1])
        except (TypeError, ValueError):
            return [0, 0]

        # Normalized coordinates (e.g. Kimi tool_use: [0.383, 0.57]).
        if (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and not (x == 0.0 and y == 0.0)
        ):
            return [int(round(x * fw)), int(round(y * fh))]

        try:
            return [
                int(round(x / self.DISPLAY_WIDTH * fw)),
                int(round(y / self.DISPLAY_HEIGHT * fh)),
            ]
        except (TypeError, ValueError):
            return [0, 0]

    def _parse_response(self, response) -> dict:
        """Find the ``aloha_action`` tool_use block and convert it to Aloha JSON."""
        if response is None:
            return {"action": "ERROR", "value": "Empty response", "position": [0, 0]}

        for block in response.content or []:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "aloha_action"
            ):
                tool_input = getattr(block, "input", None) or {}
                if isinstance(tool_input, dict):
                    return self._convert_tool_input(tool_input)

        # No tool_use block. If the model decided the task is done it'll set
        # stop_reason="end_turn"; otherwise we keep the loop alive with a
        # CONTINUE no-op (the planner will issue a new instruction next turn).
        stop_reason = getattr(response, "stop_reason", "") or ""
        if self.logger:
            self.logger.logger.warning(
                f"claude_aloha_computer_use: no aloha_action tool_use in reply "
                f"(stop_reason={stop_reason!r})"
            )
        if stop_reason == "end_turn":
            return {"action": "STOP", "value": "", "position": [0, 0]}
        return {"action": "CONTINUE", "value": "", "position": [0, 0]}

    def _convert_tool_input(self, inp: dict) -> dict:
        """Translate the tool_use input (already validated against our schema)
        into the executor's action_json contract."""
        name = str(inp.get("action") or "").upper().strip()

        position_raw = inp.get("position")
        position = self._scale_xy(position_raw) if position_raw else [0, 0]

        out: dict = {"action": name, "value": "", "position": position}

        if name == "INPUT":
            out["value"] = inp.get("text", "") or ""
        elif name == "KEY":
            out["value"] = inp.get("key", "") or ""
        elif name == "HOTKEY":
            keys = inp.get("keys") or []
            out["value"] = list(keys) if isinstance(keys, (list, tuple)) else []
        elif name == "DRAG":
            out["value"] = self._scale_xy(inp.get("drag_from"))
        elif name == "SCROLL":
            try:
                out["value"] = int(inp.get("scroll_amount", 0) or 0)
            except (TypeError, ValueError):
                out["value"] = 0
        elif name == "WAIT":
            try:
                seconds = float(inp.get("wait_seconds", 1.0) or 0.0)
            except (TypeError, ValueError):
                seconds = 1.0
            out["ms"] = int(max(0.0, seconds) * 1000)
        elif name == "STOP":
            out["value"] = inp.get("stop_summary", "") or ""

        return out
