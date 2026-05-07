import os
from openai import OpenAI

from ui_aloha.act.gui_agent.llm.llm_utils import (
    gbk_encode_decode,
    is_image_path,
    encode_image,
)


def _prepare_messages(messages: list, system: str) -> list:
    
    final_messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]}
    ]

    if isinstance(messages, str):
        final_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": gbk_encode_decode(messages)
                }]
            })
        return final_messages

    for item in messages:
        contents = []
        if isinstance(item, dict) and "content" in item:
            for cnt in item["content"]:
                if isinstance(cnt, str):
                    if is_image_path(cnt):
                        base64_image = encode_image(cnt)
                        content = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                    else:
                        content = {
                            "type": "text",
                            "text": gbk_encode_decode(cnt)
                            }
                    
                    contents.append(content)
        
            final_messages.append({"role": "user", "content": contents})
        
        elif isinstance(item, str):
            contents.append({"type": "text", "text": gbk_encode_decode(item)})
            final_messages.append({"role": "user", "content": contents})

    return final_messages


def _to_responses_input(final_messages: list) -> list:
    responses_input = []
    for msg in final_messages:
        role = msg.get("role", "user")
        contents = []
        for item in msg.get("content", []):
            if item.get("type") == "text":
                contents.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                image_url = (item.get("image_url") or {}).get("url")
                if image_url:
                    contents.append({"type": "input_image", "image_url": image_url})
        # Responses API expects a dict with role and content list
        responses_input.append({"role": role, "content": contents})
    return responses_input


def _process_responses_output(response):
    
    model = getattr(response, "model", None)
    outputs = getattr(response, "output", None)
    
    if outputs and len(outputs) > 0:
        
        # skip thinking output
        for output in outputs:
            if hasattr(output, "type") and output.type in ["thinking", "reasoning"]:
                continue
            
            # get the first content
            content = output.content
        
        if content and len(content) > 0 and hasattr(content[0], "text"):
            text = content[0].text
    
    else:
        text = ""
        
    usage = getattr(response, "usage", None)
    total_tokens = 0
    if usage is not None:
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = int(getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0))
    return text, model, total_tokens


# ---------------------------------------------------------------------------
# Anthropic / Claude routing
# ---------------------------------------------------------------------------

_IMG_EXT_TO_MEDIA = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/png",
    ".tif": "image/png",
}


def _guess_media_type(path: str) -> str:
    _, ext = os.path.splitext(path.lower())
    return _IMG_EXT_TO_MEDIA.get(ext, "image/png")


def _is_claude_model(model: str) -> bool:
    """Return True if the model name should be routed to the Anthropic API."""
    n = (model or "").lower()
    return any(tag in n for tag in ("claude", "anthropic", "opus", "sonnet", "haiku"))


def _to_anthropic_messages(messages: list) -> list:
    """Convert planner-style messages into Anthropic message blocks.

    Input format (matches `AlohaPlanner.__call__`):
        [{"role": "user", "content": [<text str>, <image_path str>, ...]}, ...]
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": [{"type": "text", "text": gbk_encode_decode(messages)}]}]

    out: list = []
    for item in messages:
        if isinstance(item, str):
            out.append({"role": "user", "content": [{"type": "text", "text": gbk_encode_decode(item)}]})
            continue
        role = item.get("role", "user")
        blocks: list = []
        for cnt in item.get("content", []):
            if not isinstance(cnt, str):
                continue
            if is_image_path(cnt):
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _guess_media_type(cnt),
                        "data": encode_image(cnt),
                    },
                })
            else:
                blocks.append({"type": "text", "text": gbk_encode_decode(cnt)})
        if blocks:
            out.append({"role": role, "content": blocks})
    return out


def _run_anthropic(
    messages: list,
    system: str,
    llm: str,
    max_tokens: int,
    temperature: float,
    api_keys: dict | None,
):
    """Call an Anthropic-compatible chat endpoint and return (text, {model: tokens}).

    Honors `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`
    and `CLAUDE_API_KEY` from env / .env, with the same Bearer-vs-x-api-key
    precedence used by ClaudeComputerUseAgent.
    """
    try:
        import anthropic
    except ImportError as e:
        return f"Error: anthropic SDK not installed ({e}).", {llm: 0}

    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or None
    api_key = (
        (api_keys or {}).get("CLAUDE_API_KEY")
        or os.environ.get("CLAUDE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or None
    )

    if not (auth_token or api_key):
        return (
            "Error: No Anthropic credentials found "
            "(set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or CLAUDE_API_KEY).",
            {llm: 0},
        )

    client_kwargs: dict = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if auth_token:
        client_kwargs["auth_token"] = auth_token
    elif api_key:
        client_kwargs["api_key"] = api_key

    client = anthropic.Anthropic(**client_kwargs)

    anth_messages = _to_anthropic_messages(messages)

    create_kwargs: dict = {
        "model": llm,
        "max_tokens": max_tokens,
        "system": system,
        "messages": anth_messages,
    }
    # Anthropic accepts temperature in [0, 1]; mirror caller intent.
    if temperature is not None:
        create_kwargs["temperature"] = max(0.0, min(1.0, float(temperature)))

    response = client.messages.create(**create_kwargs)

    # Concatenate every text block in the assistant's reply.
    text_parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text_parts.append(getattr(block, "text", ""))
    text = "".join(text_parts)

    usage = getattr(response, "usage", None)
    if usage is not None:
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        total = in_tok + out_tok
    else:
        total = 0

    return text, {llm: total}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_llm(
    messages: list,
    system: str,
    llm: str,
    max_tokens: int = 2048,
    temperature: float = 0,
    api_keys: dict | None = None,
    mode: str = "api",  # kept for compatibility; not used
    api_base: str | None = None,  # None for OpenAI API base
):
    """LLM caller that routes by model name.

    - Claude / Anthropic-compatible models (e.g. "Vendor2/Claude-4.6-Opus",
      "claude-opus-4-7-20251119") use the Anthropic SDK and respect
      ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN.
    - All other models are sent to the OpenAI-compatible Responses API.

    Returns:
        (response_text, {model_name: token_count})
    """

    if _is_claude_model(llm):
        return _run_anthropic(messages, system, llm, max_tokens, temperature, api_keys)

    api_key = None
    if api_keys and "OPENAI_API_KEY" in api_keys:
        api_key = api_keys["OPENAI_API_KEY"]
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: api_keys with OPENAI_API_KEY is required.", {llm: 0}

    client_kwargs = {}
    if api_base:
        client_kwargs["base_url"] = api_base
    if api_key:
        client_kwargs["api_key"] = api_key
        
    # Create client
    client = OpenAI(**client_kwargs)

    final_messages = _prepare_messages(messages, system)
    responses_input = _to_responses_input(final_messages)
    
    # special handling for gpt-5
    if llm.startswith("gpt-5"):
        llm_kwargs = {
            "reasoning": { "effort": "minimal" },
            "text": { "verbosity": "medium" },
        }   
    else:
        llm_kwargs = {}
    
    response = client.responses.create(
        model=llm,
        input=responses_input,
        max_output_tokens=max_tokens,
        temperature=temperature if not llm.startswith("gpt-5") else None,
        **llm_kwargs
    )

    text, model, total_tokens = _process_responses_output(response)
    token_usage_dict = {model: total_tokens}
    return text, token_usage_dict
