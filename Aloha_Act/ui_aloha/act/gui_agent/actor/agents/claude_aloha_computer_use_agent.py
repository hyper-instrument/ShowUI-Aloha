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
            #
            # thinking=disabled: some upstream vendors (gpugeek/Vendor2 for
            # Claude-4.7-Opus) inject `thinking={"type":"adaptive"}` by
            # default when they see a claude-4.7 model, which is
            # incompatible with `tool_choice={"type":"tool", ...}` and
            # produces a 400 "tool_choice 'specified' is incompatible with
            # thinking enabled" error on every step. Passing extra_body
            # explicitly overrides that inject on the way through
            # gpugeek → Anthropic. See docs/analysis/2026-07-10-human-recording-*
            # (this run's diagnostic trace).
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=[ALOHA_TOOL],
                tool_choice={"type": "tool", "name": "aloha_action"},
                extra_body={"thinking": {"type": "disabled"}},
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

        Three-branch policy (matches historical v2-tw0.3 baseline that achieved 40%
        on the 20-task set; see docs/analysis/2026-07-01-coordfix-20task-comparison.md
        §一 for the intent, and docs/analysis/2026-07-02-ext-thinking-only-20task.md
        for the failure mode observed when branch 2 was accidentally dropped):

          1. Normalized ``[0, 1]`` fractions (e.g. Kimi vendor drift) → multiply by
             actual screenshot W×H.
          2. Already-in-frame pixel coordinates (``0 ≤ x ≤ fw``, ``0 ≤ y ≤ fh``) →
             return unchanged. This is the critical fallback that lets the model emit
             raw 1920×1080 coordinates directly when it wants to.
          3. Anything else → treat as ``DISPLAY_WIDTH × DISPLAY_HEIGHT`` (1024×768)
             reference-frame integers and scale proportionally.
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

        # Branch 1: normalized coordinates (e.g. Kimi tool_use: [0.383, 0.57]).
        if (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and not (x == 0.0 and y == 0.0)
        ):
            return [int(round(x * fw)), int(round(y * fh))]

        # Branch 2: already-in-frame pixel coordinates — v2-tw0.3 baseline fallback.
        if 0.0 <= x <= float(fw) and 0.0 <= y <= float(fh):
            return [int(round(x)), int(round(y))]

        # Branch 3: 1024×768 reference-frame integers.
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

        # No aloha_action tool_use. This happens when the upstream (e.g.
        # gpugeek Vendor2) ignores `tool_choice={"type":"tool", ...}` and
        # either injects its own tools (PowerShell/computer_use) or returns
        # a plain TextBlock. Try two fallbacks before giving up.
        stop_reason = getattr(response, "stop_reason", "") or ""

        # Log a shape inventory on every degraded response — even when a
        # fallback rescues it — so post-mortems can tell which upstream
        # regression flavor we hit. See
        # docs/analysis/2026-08-16-osworld-ace-benchmark-integration-and-gap-analysis.md
        # for the taxonomy this defends against.
        if self.logger:
            block_types = [
                getattr(b, "type", type(b).__name__)
                for b in (response.content or [])
            ]
            foreign_tools = [
                getattr(b, "name", None)
                for b in (response.content or [])
                if getattr(b, "type", None) == "tool_use"
            ]
            self.logger.logger.warning(
                "claude_aloha_computer_use: degraded response "
                "(no aloha_action tool_use) "
                "stop_reason=%r block_types=%r foreign_tools=%r",
                stop_reason,
                block_types,
                foreign_tools,
            )

        # Fallback 1: parse another tool_use block (e.g. computer_use).
        # These typically carry {action: "left_click", coordinate: [x, y]}
        # or similar; map them into our Aloha action schema.
        for block in response.content or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            other_name = getattr(block, "name", None)
            other_input = getattr(block, "input", None) or {}
            if not isinstance(other_input, dict):
                continue
            mapped = self._try_map_foreign_tool(other_name, other_input)
            if mapped is not None:
                if self.logger:
                    self.logger.logger.warning(
                        "claude_aloha_computer_use: recovered action from "
                        "foreign tool_use %r (upstream ignored tool_choice)",
                        other_name,
                    )
                return mapped

        # Fallback 2: extract coordinates from any TextBlock ("click on X at
        # (123, 456)" style). Better a heuristic CLICK than a STOP.
        for block in response.content or []:
            if getattr(block, "type", None) != "text":
                continue
            text = getattr(block, "text", "") or ""
            mapped = self._try_parse_text_action(text)
            if mapped is not None:
                if self.logger:
                    self.logger.logger.warning(
                        "claude_aloha_computer_use: recovered action from "
                        "TextBlock via regex (no tool_use in reply)",
                    )
                return mapped

        # Truly nothing usable. STOP only when model explicitly ended its
        # turn; otherwise CONTINUE lets the planner retry next round.
        if self.logger:
            # Dump a bounded slice of every TextBlock so future post-mortems
            # can spot new fallback shapes we don't yet recognize.
            text_previews = []
            for b in (response.content or []):
                if getattr(b, "type", None) == "text":
                    txt = getattr(b, "text", "") or ""
                    text_previews.append(txt[:400])
            self.logger.logger.warning(
                "claude_aloha_computer_use: no rescueable action in reply "
                "(stop_reason=%r, text_previews=%r)",
                stop_reason,
                text_previews,
            )
        if stop_reason == "end_turn":
            return {"action": "STOP", "value": "", "position": [0, 0]}
        return {"action": "CONTINUE", "value": "", "position": [0, 0]}

    def _try_map_foreign_tool(self, name, inp):
        """Best-effort map from Anthropic computer_use / PowerShell / bash
        tool_use inputs to our Aloha action schema. Returns None if the
        tool call is not a UI action we can execute."""
        if not name:
            return None
        # Anthropic computer_use tool: {action: "left_click", coordinate: [x,y]}
        if isinstance(inp, dict):
            action_raw = str(inp.get("action") or "").lower().strip()
            coord = inp.get("coordinate") or inp.get("position") or inp.get("start_coordinate")
            if action_raw in ("left_click", "click", "mouse_click", "tap"):
                if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    return self._convert_tool_input({
                        "action": "CLICK", "position": [coord[0], coord[1]],
                    })
            if action_raw in ("right_click", "mouse_right_click"):
                if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    return self._convert_tool_input({
                        "action": "RIGHT_CLICK", "position": [coord[0], coord[1]],
                    })
            if action_raw in ("double_click",):
                if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    return self._convert_tool_input({
                        "action": "DOUBLE_CLICK", "position": [coord[0], coord[1]],
                    })
            if action_raw in ("type", "type_text", "input"):
                text = inp.get("text") or inp.get("value") or ""
                if text:
                    return self._convert_tool_input({
                        "action": "INPUT", "text": text,
                    })
            if action_raw in ("key", "hotkey", "press_key"):
                key = inp.get("text") or inp.get("key") or ""
                if key:
                    # Anthropic uses "+", we use lists.
                    keys = [k.strip() for k in key.replace(" ", "").split("+") if k.strip()]
                    if len(keys) == 1:
                        return self._convert_tool_input({
                            "action": "KEY", "key": keys[0],
                        })
                    if len(keys) > 1:
                        return self._convert_tool_input({
                            "action": "HOTKEY", "keys": keys,
                        })
            if action_raw in ("scroll",):
                if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    amt = inp.get("scroll_amount", 0) or inp.get("clicks", 0)
                    direction = str(inp.get("scroll_direction") or "").lower()
                    try:
                        amt = int(amt)
                    except (TypeError, ValueError):
                        amt = 3
                    if direction == "down":
                        amt = -abs(amt)
                    elif direction == "up":
                        amt = abs(amt)
                    return self._convert_tool_input({
                        "action": "SCROLL", "position": [coord[0], coord[1]],
                        "scroll_amount": amt,
                    })
            # bash / PowerShell etc. — not a UI action, treat as CONTINUE.
            if name.lower() in ("bash", "powershell", "shell", "cmd"):
                return {"action": "CONTINUE", "value": "", "position": [0, 0]}
        return None

    def _try_parse_text_action(self, text):
        """Regex-parse a plain-language actor reply for the common shapes.
        Returns an action dict or None."""
        import re
        if not text:
            return None

        # First: pyautogui code block. Vendor2/gpugeek routes sometimes drop
        # tool_use entirely and instead emit a python snippet — see
        # docs/analysis/2026-08-16-osworld-ace-benchmark-integration-and-gap-analysis.md
        # §3.2. The block may pass coordinates as literal digits OR via named
        # variables assigned a few lines above (e.g. `start_x, start_y = 651, 395`
        # then `pyautogui.moveTo(start_x, start_y)`). The bare-tuple regex
        # below only catches the literal-digit case; this branch also handles
        # variable indirection and the moveTo→mouseDown→moveTo→mouseUp drag
        # pattern.
        pg = self._parse_pyautogui_block(text)
        if pg is not None:
            return pg

        # "at (x, y)" / "at coordinates (x, y)" / "coordinates: (x, y)"
        m = re.search(r"(?i)(?:at\s+(?:coordinates?\s+)?|coordinates?[:\s]+)\(?(\d{1,4})[,\s]+(\d{1,4})\)?", text)
        if not m:
            m = re.search(r"\((\d{1,4})[,\s]+(\d{1,4})\)", text)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            low = text.lower()
            if "right-click" in low or "right click" in low or "context menu" in low:
                return self._convert_tool_input({"action": "RIGHT_CLICK", "position": [x, y]})
            if "double-click" in low or "double click" in low:
                return self._convert_tool_input({"action": "DOUBLE_CLICK", "position": [x, y]})
            # default: single click
            return self._convert_tool_input({"action": "CLICK", "position": [x, y]})
        # "type 'foo'" / "type \"foo\""
        m = re.search(r"(?i)\btype\s+['\"]([^'\"]{1,200})['\"]", text)
        if m:
            return self._convert_tool_input({"action": "INPUT", "text": m.group(1)})
        # "press <key>" — single alnum key or common named keys
        m = re.search(r"(?i)\bpress\s+(?:the\s+)?['\"]?([A-Za-z][A-Za-z0-9_+\-]{0,20})['\"]?\s*key", text)
        if m:
            key = m.group(1)
            if "+" in key:
                keys = [k.strip() for k in key.split("+") if k.strip()]
                return self._convert_tool_input({"action": "HOTKEY", "keys": keys})
            return self._convert_tool_input({"action": "KEY", "key": key})
        return None

    def _parse_pyautogui_block(self, text):
        """Extract an Aloha action from a pyautogui code snippet.

        Handles three shapes:
          1. Direct-digit args: ``pyautogui.click(651, 395)``
          2. Variable-mediated args: ``start_x, start_y = 651, 395`` on one line,
             ``pyautogui.moveTo(start_x, start_y)`` on another
          3. Drag sandwich: ``moveTo(start)`` → ``mouseDown`` → ``moveTo(end)``
             → ``mouseUp``, or ``moveTo(start)`` → ``dragTo(end, ...)``

        Also recognises ``typewrite``/``write``/``press``/``hotkey``/``scroll``.
        Returns an action dict (already run through :meth:`_convert_tool_input`)
        or ``None`` if no pyautogui call is present.
        """
        import re

        # 1) Build variable → digit map from assignments in this text block.
        var_map: dict[str, int] = {}
        # Tuple assignment: "a, b = 1, 2"
        for m in re.finditer(
            r"\b(\w+)\s*,\s*(\w+)\s*=\s*(\d{1,4})\s*,\s*(\d{1,4})\b", text
        ):
            var_map[m.group(1)] = int(m.group(3))
            var_map[m.group(2)] = int(m.group(4))
        # Scalar assignment on its own line: "x = 42"
        for m in re.finditer(
            r"(?m)^\s*(\w+)\s*=\s*(\d{1,4})\s*$", text
        ):
            var_map.setdefault(m.group(1), int(m.group(2)))

        def resolve(tok: str):
            tok = tok.strip()
            if re.fullmatch(r"-?\d{1,4}", tok):
                return int(tok)
            # Strip potential type coercions like int(start_x).
            im = re.fullmatch(r"int\(\s*(\w+)\s*\)", tok)
            if im:
                tok = im.group(1)
            return var_map.get(tok)

        def pos_from(args: list[str]):
            if len(args) < 2:
                return None
            x = resolve(args[0])
            y = resolve(args[1])
            if x is None or y is None:
                return None
            return [x, y]

        # 2) Collect pyautogui.<func>(...) calls in source order.
        calls: list[tuple[str, list[str]]] = []
        for m in re.finditer(r"pyautogui\.(\w+)\s*\(([^)]*)\)", text):
            func = m.group(1)
            raw = m.group(2)
            args = [a.strip() for a in raw.split(",")] if raw.strip() else []
            # Drop kwargs like button='left' / duration=0.5 for positional lookup.
            positional = [a for a in args if "=" not in a]
            calls.append((func, positional))

        if not calls:
            return None

        # 3) Drag detection has priority over lone clicks.
        # 3a) dragTo(...) preceded by moveTo(...).
        for i, (func, args) in enumerate(calls):
            if func == "dragTo":
                end = pos_from(args)
                if end is None:
                    continue
                start = None
                for pfunc, pargs in reversed(calls[:i]):
                    if pfunc == "moveTo":
                        start = pos_from(pargs)
                        if start:
                            break
                if start:
                    return self._convert_tool_input({
                        "action": "DRAG",
                        "drag_from": start,
                        "position": end,
                    })

        # 3b) moveTo → mouseDown → moveTo → mouseUp sandwich.
        move_calls = [args for f, args in calls if f == "moveTo"]
        has_down = any(f == "mouseDown" for f, _ in calls)
        has_up = any(f == "mouseUp" for f, _ in calls)
        if has_down and has_up and len(move_calls) >= 2:
            start = pos_from(move_calls[0])
            end = pos_from(move_calls[-1])
            if start and end:
                return self._convert_tool_input({
                    "action": "DRAG",
                    "drag_from": start,
                    "position": end,
                })

        # 4) Otherwise map the first click-like call.
        click_map = {
            "click": "CLICK",
            "leftClick": "CLICK",
            "moveTo": "CLICK",  # bare moveTo without buttons → click target
            "rightClick": "RIGHT_CLICK",
            "doubleClick": "DOUBLE_CLICK",
            "tripleClick": "TRIPLE_CLICK",
        }
        for func, args in calls:
            if func in click_map:
                pos = pos_from(args)
                if pos:
                    return self._convert_tool_input({
                        "action": click_map[func],
                        "position": pos,
                    })

        # 5) scroll(amt) or scroll(amt, x, y).
        for func, args in calls:
            if func == "scroll" and args:
                amt = resolve(args[0])
                if amt is None:
                    continue
                pos = pos_from(args[1:3]) if len(args) >= 3 else None
                return self._convert_tool_input({
                    "action": "SCROLL",
                    "position": pos or [0, 0],
                    "scroll_amount": int(amt),
                })

        # 6) typewrite('foo') / write('foo').
        for func, args in calls:
            if func in ("typewrite", "write") and args:
                raw = args[0].strip()
                if len(raw) >= 2 and raw[0] in "'\"" and raw[0] == raw[-1]:
                    return self._convert_tool_input({
                        "action": "INPUT",
                        "text": raw[1:-1],
                    })

        # 7) press('enter') / hotkey('ctrl', 's').
        for func, args in calls:
            if func == "press" and args:
                raw = args[0].strip()
                if len(raw) >= 2 and raw[0] in "'\"" and raw[0] == raw[-1]:
                    return self._convert_tool_input({
                        "action": "KEY",
                        "key": raw[1:-1],
                    })
            if func == "hotkey" and args:
                keys: list[str] = []
                for a in args:
                    a = a.strip()
                    if len(a) >= 2 and a[0] in "'\"" and a[0] == a[-1]:
                        keys.append(a[1:-1])
                if keys:
                    return self._convert_tool_input({
                        "action": "HOTKEY",
                        "keys": keys,
                    })

        return None

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
