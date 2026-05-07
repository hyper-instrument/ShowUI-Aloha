import os
import re

from jinja2 import Environment, FileSystemLoader

from ui_aloha.act.utils.path_utils import prompt_templates_path
from ui_aloha.act.gui_agent.llm.llm_utils import encode_image

try:
    import anthropic
    from anthropic.types.beta import BetaTextBlock, BetaToolUseBlock
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    BetaTextBlock = None
    BetaToolUseBlock = None


_VERSION_RE = re.compile(r"(\d+)[.\-_](\d+)")
_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def _resolve_tool_version(model: str) -> tuple[str, str]:
    """Return (computer_tool_type, beta_flag) for a given model name.

    Per Anthropic computer use docs:
      - computer-use-2025-11-24: Opus 4.5/4.6/4.7, Sonnet 4.6
      - computer-use-2025-01-24: Sonnet 4.5, Haiku 4.5, Opus 4.1, Sonnet 4, Opus 4, Sonnet 3.7

    Robust to both Anthropic-native names ("claude-opus-4-7-...") and
    third-party vendor names ("Vendor2/Claude-4.7-Opus") where the
    family token may appear before or after the version number.
    """
    n = (model or "").lower()
    if "opus" in n:
        family = "opus"
    elif "haiku" in n:
        family = "haiku"
    elif "sonnet" in n:
        family = "sonnet"
    else:
        family = ""
    m = _VERSION_RE.search(n)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2)) if m else 0

    needs_v2 = (
        (family == "opus" and (major, minor) >= (4, 5))
        or (family == "sonnet" and (major, minor) >= (4, 6))
    )
    if needs_v2:
        return "computer_20251124", "computer-use-2025-11-24"
    return "computer_20250124", "computer-use-2025-01-24"


class ClaudeComputerUseAgent:
    def __init__(
        self,
        api_key: str | None = None,
        logger=None,
        base_url: str | None = None,
        auth_token: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ):
        # Resolve credentials/endpoint with env fallback so this works against
        # both api.anthropic.com and OpenAI-compatible Anthropic vendors
        # (e.g. gpugeek) that authenticate with `Authorization: Bearer ...`.
        base_url = base_url or os.getenv("ANTHROPIC_BASE_URL") or None
        auth_token = auth_token or os.getenv("ANTHROPIC_AUTH_TOKEN") or None
        api_key = api_key or os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or None

        client_kwargs: dict = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        # Bearer auth wins when set (vendor case); otherwise use x-api-key.
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

        # Display the agent advertises to the model. Coordinates returned
        # by the model are in this space and we rescale to the executor's
        # 1920x1080 reference frame below.
        self.DISPLAY_WIDTH = 1024
        self.DISPLAY_HEIGHT = 768
        # Reference frame used by AlohaExecutor for coordinate scaling.
        self.TARGET_WIDTH = 1920
        self.TARGET_HEIGHT = 1080

    def execute(self, instruction, screenshot_path, system_prompt, logging_dir):
        """Execute Claude Computer Use agent action"""

        if not ANTHROPIC_AVAILABLE or not self.client:
            error_msg = "Anthropic library not available or no credentials provided"
            if self.logger:
                self.logger.logger.error(error_msg)
            return {"action": "ERROR", "value": error_msg, "position": [0, 0]}, False

        screenshot_base64 = encode_image(screenshot_path)

        try:
            # Render user instruction template via Jinja2
            templates_dir = prompt_templates_path()
            env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            user_text = env.get_template("actor/user_cua.txt").render(task=instruction)

            tool_type, beta_flag = _resolve_tool_version(self.model)
            if self.logger:
                self.logger.logger.info(
                    f"claude_computer_use: model={self.model} "
                    f"base_url={self.base_url or 'default'} "
                    f"tool={tool_type} beta={beta_flag}"
                )

            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                tools=[{
                    "type": tool_type,
                    "name": "computer",
                    "display_width_px": self.DISPLAY_WIDTH,
                    "display_height_px": self.DISPLAY_HEIGHT,
                    "display_number": 1,
                }],
                system=system_prompt,
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
                        }
                    ]
                }],
                betas=[beta_flag],
            )

            # Log raw response
            if self.logger:
                self.logger.log_json({"response": str(response)}, "actor_claude_computer_use_raw_response.json", logging_dir)

            action_json = self._parse_response(response)

            # Log parsed action
            if self.logger:
                self.logger.log_json(action_json, "actor_claude_computer_use_parsed_action.json", logging_dir)

            return action_json, action_json.get("action") == "STOP"

        except Exception as e:
            error_msg = f"Error processing claude-computer-use response: {e}"
            if self.logger:
                self.logger.logger.error(error_msg)
                self.logger.log_error(e, {"mode": "claude-computer-use"}, target_dir=logging_dir)

            return {"action": "ERROR", "value": str(e), "position": [0, 0]}, False

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _scale_xy(self, coord) -> list[int]:
        """Scale a [x, y] from (DISPLAY_WIDTH, DISPLAY_HEIGHT) into the
        executor's (TARGET_WIDTH, TARGET_HEIGHT) reference frame."""
        if not coord or len(coord) < 2:
            return [0, 0]
        return [
            int(coord[0] / self.DISPLAY_WIDTH * self.TARGET_WIDTH),
            int(coord[1] / self.DISPLAY_HEIGHT * self.TARGET_HEIGHT),
        ]

    def _parse_response(self, response):
        """Convert a Claude computer-use response into Aloha's action_json.

        Strategy:
          - First text block is treated as reasoning and logged.
          - First tool_use block is converted via `_convert_tool_use_to_action`.
          - If no tool_use is present:
              * `stop_reason == "end_turn"`  -> STOP (task complete from Claude's view)
              * otherwise                    -> CONTINUE (no-op; loop fetches next frame)
        """
        if response is None:
            return {"action": "ERROR", "value": "Empty response", "position": [0, 0]}

        text_chunks: list[str] = []
        tool_use_input = None

        for item in response.content or []:
            if isinstance(item, BetaTextBlock):
                text_chunks.append(item.text or "")
                if self.logger:
                    self.logger.logger.info(f"claude_computer_use: reasoning={item.text}")
            elif isinstance(item, BetaToolUseBlock):
                if tool_use_input is None:
                    tool_use_input = item.input

        if tool_use_input is None:
            stop_reason = getattr(response, "stop_reason", "") or ""
            joined = " ".join(t for t in text_chunks if t).strip()
            if stop_reason == "end_turn":
                if self.logger:
                    self.logger.logger.info("claude_computer_use: end_turn -> STOP")
                return {"action": "STOP", "value": joined, "position": [0, 0]}
            if self.logger:
                self.logger.logger.info(
                    f"claude_computer_use: no tool_use, stop_reason={stop_reason!r} -> CONTINUE"
                )
            return {"action": "CONTINUE", "value": joined, "position": [0, 0]}

        action_json = self._convert_tool_use_to_action(tool_use_input)
        if action_json is None:
            if self.logger:
                self.logger.logger.info(
                    f"claude_computer_use: unsupported action_type={tool_use_input.get('action')!r}"
                    f" input={tool_use_input} -> CONTINUE"
                )
            return {"action": "CONTINUE", "value": "", "position": [0, 0]}
        return action_json

    # Aloha high-level action names accepted by AlohaExecutor.supported_actions.
    _ALOHA_HIGH_LEVEL_ACTIONS = frozenset({
        "CLICK", "RIGHT_CLICK", "INPUT", "MOVE", "HOVER", "ENTER",
        "ESC", "ESCAPE", "PRESS", "KEY", "HOTKEY", "DRAG", "SCROLL",
        "DOUBLE_CLICK", "TRIPLE_CLICK", "WAIT", "PAUSE", "CONTINUE", "STOP",
    })

    # Vendors are inconsistent about which key holds the action name. Accept
    # any of these (first non-empty wins).
    _ACTION_NAME_KEYS = ("action", "action_type", "type", "name")
    # Likewise for click coordinates.
    _POSITION_KEYS = ("position", "coordinate", "coord", "xy")

    @staticmethod
    def _first_non_empty(d: dict, keys):
        for k in keys:
            v = d.get(k)
            if v not in (None, ""):
                return v
        return None

    def _normalize_vendor_action(self, nested: dict) -> dict | None:
        """Some Anthropic-compatible vendors (gpugeek today) pre-translate the
        Computer Use tool call into Aloha's high-level format and ship it as
        ``{"action": {"action": "CLICK", "position": [...], ...}}`` — or the
        equivalent shape with ``action_type`` / ``coordinate`` keys, since the
        vendor isn't consistent. Coordinates are still in the model's 1024x768
        display space, so we rescale and forward to the executor.
        """
        name_raw = self._first_non_empty(nested, self._ACTION_NAME_KEYS)
        name = str(name_raw or "").upper().strip()
        if not name:
            return None
        if name not in self._ALOHA_HIGH_LEVEL_ACTIONS:
            return None  # caller maps to CONTINUE no-op

        position = self._first_non_empty(nested, self._POSITION_KEYS)

        out: dict = {
            "action": name,
            "value": nested.get("value", nested.get("text", "")),
            "position": self._scale_xy(position) if position else [0, 0],
        }
        # Carry-over and rescale optional drag/wait/text fields the executor
        # parsers know about.
        for k in ("from", "to", "start", "end"):
            v = nested.get(k)
            if v is not None:
                out[k] = self._scale_xy(v)
        for k in ("ms", "text"):
            if k in nested and k not in out:
                out[k] = nested[k]
        return out

    def _convert_tool_use_to_action(self, tool_input: dict) -> dict | None:
        """Translate Claude's `computer` tool_use input into an Aloha action.

        Claude's standard (Anthropic) action vocabulary:
          screenshot, left_click, right_click, middle_click, double_click,
          triple_click, mouse_move, left_mouse_down/up, left_click_drag,
          key, hold_key, type, scroll, wait, cursor_position, zoom

        Returns None for actions we deliberately don't translate, so the caller
        can map them to a CONTINUE no-op.
        """
        raw_action = tool_input.get("action")

        # Vendor quirk #1 (gpugeek nested): the action is a dict shaped like
        # Aloha's action_json, e.g.
        #   {"action": {"action": "CLICK", "position": [...]}}
        # or sometimes with the inner key spelled "action_type":
        #   {"action": {"action_type": "CLICK", "position": [...]}}
        if isinstance(raw_action, dict):
            if self.logger:
                self.logger.logger.info(
                    f"claude_computer_use: vendor-nested action={raw_action}"
                )
            return self._normalize_vendor_action(raw_action)

        # Vendor quirk #2 (gpugeek flat): the *whole* tool_input is already in
        # Aloha format with UPPERCASE action names. Detect by either `action`
        # or any of the alias keys carrying a known Aloha high-level action.
        flat_name = self._first_non_empty(tool_input, self._ACTION_NAME_KEYS)
        if (
            isinstance(flat_name, str)
            and flat_name.upper().strip() in self._ALOHA_HIGH_LEVEL_ACTIONS
        ):
            if self.logger:
                self.logger.logger.info(
                    f"claude_computer_use: vendor-flat action={tool_input}"
                )
            return self._normalize_vendor_action(tool_input)

        action_type = (raw_action or "").lower()
        if self.logger:
            self.logger.logger.info(
                f"claude_computer_use: action_type={action_type} input={tool_input}"
            )

        # Client-side observations that don't require executor work — the next
        # loop iteration will already supply a fresh screenshot.
        if action_type in ("screenshot", "cursor_position", "zoom"):
            return {"action": "CONTINUE", "value": "", "position": [0, 0]}

        # Fine-grained mouse-button events aren't exposed by the Aloha executor.
        # Treat them as no-ops; Claude rarely uses these without a follow-up.
        if action_type in ("left_mouse_down", "left_mouse_up"):
            return {"action": "CONTINUE", "value": "", "position": [0, 0]}

        if action_type == "left_click":
            return {
                "action": "CLICK",
                "value": "",
                "position": self._scale_xy(tool_input.get("coordinate")),
            }

        if action_type == "right_click":
            return {
                "action": "RIGHT_CLICK",
                "value": "",
                "position": self._scale_xy(tool_input.get("coordinate")),
            }

        if action_type == "middle_click":
            # Aloha executor has no middle-click; fall back to a left-click at
            # the same point. Better to act than to skip.
            return {
                "action": "CLICK",
                "value": "",
                "position": self._scale_xy(tool_input.get("coordinate")),
            }

        if action_type == "double_click":
            return {
                "action": "DOUBLE_CLICK",
                "value": "",
                "position": self._scale_xy(tool_input.get("coordinate")),
            }

        if action_type == "triple_click":
            return {
                "action": "TRIPLE_CLICK",
                "value": "",
                "position": self._scale_xy(tool_input.get("coordinate")),
            }

        if action_type == "mouse_move":
            return {
                "action": "MOVE",
                "value": "",
                "position": self._scale_xy(tool_input.get("coordinate")),
            }

        if action_type == "left_click_drag":
            start = tool_input.get("start_coordinate") or tool_input.get("from")
            end = tool_input.get("coordinate") or tool_input.get("to")
            return {
                "action": "DRAG",
                # Aloha's _parse_drag accepts (value=start, position=end).
                "value": self._scale_xy(start),
                "position": self._scale_xy(end),
            }

        if action_type in ("key", "hold_key", "keypress"):
            text = tool_input.get("text") or tool_input.get("key") or ""
            # Anthropic uses xdotool-style combos like "ctrl+s" / "cmd+shift+t".
            # Aloha's _parse_key_or_hotkey accepts a list and presses each key in
            # sequence (close enough for chord-style shortcuts under pyautogui).
            if isinstance(text, str) and "+" in text:
                value = [k.strip() for k in text.split("+") if k.strip()]
            else:
                value = text
            return {"action": "KEY", "value": value, "position": [0, 0]}

        if action_type == "type":
            return {
                "action": "INPUT",
                "value": tool_input.get("text", ""),
                "position": [0, 0],
            }

        if action_type == "scroll":
            direction = (tool_input.get("scroll_direction") or "down").lower()
            try:
                amount = int(tool_input.get("scroll_amount") or 1)
            except (TypeError, ValueError):
                amount = 1
            if direction in ("up", "left"):
                value = -amount
            else:  # "down" / "right" / unknown
                value = amount
            coord = tool_input.get("coordinate")
            return {
                "action": "SCROLL",
                "value": value,
                "position": self._scale_xy(coord) if coord else [0, 0],
            }

        if action_type == "wait":
            try:
                seconds = float(tool_input.get("duration") or 1.0)
            except (TypeError, ValueError):
                seconds = 1.0
            return {
                "action": "WAIT",
                "value": "",
                "position": [0, 0],
                "ms": int(max(0.0, seconds) * 1000),
            }

        return None
