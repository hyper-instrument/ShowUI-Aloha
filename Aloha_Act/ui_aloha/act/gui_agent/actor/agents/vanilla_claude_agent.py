"""Vanilla Claude actor: same LLM + tool schema as ``ClaudeAlohaComputerUseAgent``,
but driven *without* the Aloha trajectory / planner.

Why this exists
---------------
``ClaudeAlohaComputerUseAgent`` is normally invoked downstream of
``AlohaPlanner``, which loads a learned trace (see ``TrajectoryManager``) and
embeds it as in-context guidance. That gives the actor strong prior knowledge
of the recorded workflow, but couples every run to a specific trace.
# 
Sometimes we want to evaluate the model's *raw* computer-use ability: no trace,
no planner, no in-context examples. Just (system prompt + ``aloha_action`` tool
+ current screenshot + accumulated action history) → next action. That's what
this agent does.

It deliberately reuses ``ALOHA_TOOL`` and the parsing helpers from
``claude_aloha_computer_use_agent`` so the action contract is identical and
the executor consumes its output unchanged.
"""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader

from ui_aloha.act.gui_agent.llm.llm_utils import encode_image
from ui_aloha.act.utils.path_utils import prompt_templates_path

from .claude_aloha_computer_use_agent import (
    ANTHROPIC_AVAILABLE,
    ALOHA_TOOL,
    ClaudeAlohaComputerUseAgent,
)


class VanillaClaudeAgent(ClaudeAlohaComputerUseAgent):
    """Plain Claude actor that ignores the Aloha trajectory.

    Inherits the ``aloha_action`` tool schema, response parsing, and
    coordinate scaling from :class:`ClaudeAlohaComputerUseAgent`; only
    overrides :meth:`execute` to render a vanilla user prompt
    (task + action history, no planner output, no trajectory examples).
    """

    def __init__(
        self,
        api_key: str | None = None,
        logger=None,
        base_url: str | None = None,
        auth_token: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ):
        super().__init__(
            api_key=api_key,
            logger=logger,
            base_url=base_url,
            auth_token=auth_token,
            model=model,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def execute(
        self,
        instruction,
        screenshot_path,
        system_prompt,
        logging_dir,
        action_history=None,
    ):
        """Send one user turn (task + history + screenshot) and return
        ``(action_json, complete_flag)``.

        Args:
            instruction: The raw task string (NOT a planner output dict).
            screenshot_path: Path to the current screenshot.
            system_prompt: Vanilla system prompt (no trajectory context).
            logging_dir: Per-request log directory.
            action_history: Optional list of strings or dicts describing
                actions already executed this episode. Rendered into the
                user prompt so the model has continuity across stateless
                server requests.
        """
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

            task_str = self._coerce_task(instruction)
            history_str = self._format_history(action_history)

            templates_dir = prompt_templates_path()
            env = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                autoescape=False,
                trim_blocks=True,
                lstrip_blocks=True,
            )
            user_text = env.get_template("actor/user_cua_vanilla.txt").render(
                task=task_str,
                action_history_str=history_str,
                history_length=len(action_history or []),
            )

            if self.logger:
                self.logger.logger.info(
                    f"vanilla_claude: model={self.model} "
                    f"base_url={self.base_url or 'default'} "
                    f"history_len={len(action_history or [])} "
                    "(no trajectory, custom tool use)"
                )

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
                    "actor_vanilla_claude_raw_response.json",
                    logging_dir,
                )

            action_json = self._parse_response(response)

            if self.logger:
                self.logger.log_json(
                    action_json,
                    "actor_vanilla_claude_parsed_action.json",
                    logging_dir,
                )

            return action_json, action_json.get("action") == "STOP"

        except Exception as e:
            error_msg = f"Error processing vanilla-claude response: {e}"
            if self.logger:
                self.logger.logger.error(error_msg)
                self.logger.log_error(
                    e, {"mode": "vanilla-claude"}, target_dir=logging_dir
                )
            return {"action": "ERROR", "value": str(e), "position": [0, 0]}, False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_task(instruction) -> str:
        """In normal Aloha runs the actor receives the full planner dict; here
        we want the raw user task instead. Accept either shape so the agent
        also degrades gracefully if some caller still passes a dict."""
        if isinstance(instruction, dict):
            for key in ("task", "query", "instruction", "user_task"):
                v = instruction.get(key)
                if isinstance(v, str) and v.strip():
                    return v
            return str(instruction)
        return str(instruction or "")

    @staticmethod
    def _format_history(action_history) -> str:
        """Render an action_history list into a readable bulleted string."""
        if not action_history:
            return "(none — this is the first action.)"
        lines: list[str] = []
        for i, item in enumerate(action_history, start=1):
            if isinstance(item, dict):
                text = str(item)
            else:
                text = str(item).rstrip()
            lines.append(f"  {i}. {text}")
        return "\n".join(lines)
