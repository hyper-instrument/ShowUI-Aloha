"""Actor orchestrator that routes to concrete backend agents."""

from jinja2 import Environment, FileSystemLoader
from ui_aloha.act.utils.path_utils import prompt_templates_path
from ui_aloha.act.utils.logger_utils import LoggerUtils

# Import the separate agent modules
from ui_aloha.act.gui_agent.actor.agents import (
    OAIOperatorAgent,
    ClaudeComputerUseAgent,
    ClaudeAlohaComputerUseAgent,
    VanillaClaudeAgent,
    UITarsAgent,
)

class AlohaActor:
    """High-level actor that selects and executes a specific agent backend."""

    def __init__(
        self,
        api_keys: dict | None = None,
        model: str = "oai-operator",
        os_name: str = "windows",
        claude_model: str | None = None,
    ):
        self.api_keys = api_keys
        self.model = model
        self.os_name = os_name

        # Initialize logger
        self.logger = LoggerUtils(component_name="actor")

        # Extract API keys / endpoints
        if api_keys:
            operator_openai_api_key = api_keys.get("OPERATOR_OPENAI_API_KEY") or api_keys.get("OPENAI_API_KEY", "")
            claude_api_key = api_keys.get("CLAUDE_API_KEY", "")
            anthropic_base_url = api_keys.get("ANTHROPIC_BASE_URL") or None
            anthropic_auth_token = api_keys.get("ANTHROPIC_AUTH_TOKEN") or None
        else:
            operator_openai_api_key = ""
            claude_api_key = ""
            anthropic_base_url = None
            anthropic_auth_token = None

        # Initialize agent modules
        self.oai_operator_agent = OAIOperatorAgent(
            api_key=operator_openai_api_key,
            logger=self.logger
        )

        self.claude_computer_use_agent = ClaudeComputerUseAgent(
            api_key=claude_api_key,
            logger=self.logger,
            base_url=anthropic_base_url,
            auth_token=anthropic_auth_token,
            model=claude_model,
        )

        self.claude_aloha_computer_use_agent = ClaudeAlohaComputerUseAgent(
            api_key=claude_api_key,
            logger=self.logger,
            base_url=anthropic_base_url,
            auth_token=anthropic_auth_token,
            model=claude_model,
        )

        # Vanilla Claude actor: same LLM/tool capability as
        # ClaudeAlohaComputerUseAgent, but invoked without any trajectory /
        # planner context so we can evaluate raw computer-use behavior.
        self.vanilla_claude_agent = VanillaClaudeAgent(
            api_key=claude_api_key,
            logger=self.logger,
            base_url=anthropic_base_url,
            auth_token=anthropic_auth_token,
            model=claude_model,
        )

        self.ui_tars_agent = UITarsAgent(
            logger=self.logger
        )

        # Jinja2 template environment
        templates_dir = prompt_templates_path()
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Define system prompts via Jinja2
        self.oai_operator_system_prompt = self._jinja_env.get_template(
            "actor/system_cua.txt").render(os_name=self.os_name)
        self.claude_cua_system_prompt = self._jinja_env.get_template(
            "actor/system_cua.txt").render(os_name=self.os_name)
        # Aloha-style Claude actor uses a stricter prompt that bakes in the
        # Aloha JSON output contract (no Anthropic tool_use scaffolding).
        self.claude_aloha_cua_system_prompt = self._jinja_env.get_template(
            "actor/system_cua_aloha.txt").render(os_name=self.os_name)
        # Vanilla actor uses a separate prompt that explicitly tells the model
        # there is no demonstration / trajectory available.
        self.vanilla_claude_system_prompt = self._jinja_env.get_template(
            "actor/system_cua_vanilla.txt").render(os_name=self.os_name)
        self.uitars_grounding_system_prompt = self._jinja_env.get_template(
            "actor/system_ui_tars.txt").render()


    def __call__(
        self,
        mode: str | None = None,
        messages: str | dict = "",
        screenshot_path: str = "",
        logging_dir: str = ".cache/",
        action_history: list | None = None,
    ):
        """Execute the selected agent and return its next action.

        Args:
            mode: Optional override; one of "oai-operator",
                "claude-computer-use", "claude-aloha-computer-use",
                "vanilla-claude", "ui-tars".
            messages: Planner output (dict) for trajectory-driven modes, or a
                raw task string for vanilla mode.
            screenshot_path: Path to the current UI screenshot.
            logging_dir: Directory to store logs.
            action_history: Optional list of prior actions; only used by the
                vanilla agent (which has no planner / trajectory context).

        Returns:
            (action_dict_wrapped, complete_flag)
        """

        # Ensure task is properly formatted
        if isinstance(messages, dict):
            task = messages
        else:
            task = messages

        effective_mode = (mode or self.model)
        self.logger.logger.info(f"AlohaActor Mode: {effective_mode}")

        # -------------------------------
        # Execute the appropriate agent based on mode
        # -------------------------------
        if effective_mode == "oai-operator":
            response, complete_flag = self.oai_operator_agent.execute(
                instruction=task,
                screenshot_path=screenshot_path,
                os_name=self.os_name,
                system_prompt=self.oai_operator_system_prompt,
                logging_dir=logging_dir
            )
        
        elif effective_mode == "claude-computer-use":
            response, complete_flag = self.claude_computer_use_agent.execute(
                instruction=task,
                screenshot_path=screenshot_path,
                system_prompt=self.claude_cua_system_prompt,
                logging_dir=logging_dir
            )

        elif effective_mode == "claude-aloha-computer-use":
            response, complete_flag = self.claude_aloha_computer_use_agent.execute(
                instruction=task,
                screenshot_path=screenshot_path,
                system_prompt=self.claude_aloha_cua_system_prompt,
                logging_dir=logging_dir
            )

        elif effective_mode == "vanilla-claude":
            response, complete_flag = self.vanilla_claude_agent.execute(
                instruction=task,
                screenshot_path=screenshot_path,
                system_prompt=self.vanilla_claude_system_prompt,
                logging_dir=logging_dir,
                action_history=action_history,
            )

        elif effective_mode == "ui-tars":  # qwen related
            response, complete_flag = self.ui_tars_agent.execute(
                instruction=task,
                screenshot_path=screenshot_path,
                system_prompt=self.uitars_grounding_system_prompt,
                logging_dir=logging_dir
            )
        
        else:
            error_msg = f"Invalid mode for AlohaActor: {effective_mode}"
            self.logger.logger.error(error_msg)
            response = {"action": "ERROR", "value": error_msg, "position": [0, 0]}
            complete_flag = False
        
        
        self.logger.log_json(response, f"actor_{effective_mode}_action.json", logging_dir)

        # Return in the original format for backward compatibility
        final_response = {"content": response, "role": "assistant"}
        return final_response, complete_flag
