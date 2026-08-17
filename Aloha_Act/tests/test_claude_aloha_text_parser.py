"""Unit tests for ClaudeAlohaComputerUseAgent._try_parse_text_action.

Motivated by the 2026-08-13 ace-benchmark 20-task run (see
docs/analysis/2026-08-16-osworld-ace-benchmark-integration-and-gap-analysis.md
§3.2): gpugeek/Vendor2 routed responses drop tool_use and emit a python
snippet with pyautogui calls, sometimes with variable-mediated coordinates.
The pre-fix regex only caught bare tuples like ``(651, 395)``, missing
``pyautogui.moveTo(start_x, start_y)``, and every such task went STOP→DONE
at step 1.

These tests exercise the pyautogui-block fallback branch. They do NOT hit
the Anthropic API — ``_try_parse_text_action`` is a pure text function.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The agent module lives under ui_aloha/act/gui_agent/actor/agents/. That
# package is importable when the Aloha_Act root is on sys.path.
ALOHA_ACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALOHA_ACT_ROOT))

from ui_aloha.act.gui_agent.actor.agents.claude_aloha_computer_use_agent import (  # noqa: E402
    ClaudeAlohaComputerUseAgent,
)


def _agent():
    """Build a bare agent without credentials — we only test the parser."""
    a = ClaudeAlohaComputerUseAgent.__new__(ClaudeAlohaComputerUseAgent)
    a.logger = None
    a.DISPLAY_WIDTH = 1024
    a.DISPLAY_HEIGHT = 768
    a.TARGET_WIDTH = 1920
    a.TARGET_HEIGHT = 1080
    # In-frame pixel coordinates should pass through _scale_xy unchanged
    # (branch 2 of the three-branch policy).
    a._executor_frame_w = 1920
    a._executor_frame_h = 1080
    return a


# --- literal-digit pyautogui calls ------------------------------------------------

def test_moveto_literal_digits_maps_to_click():
    text = "```python\npyautogui.moveTo(767, 524)\n```"
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "CLICK", "value": "", "position": [767, 524]}


def test_click_literal_digits():
    text = "pyautogui.click(100, 200)"
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "CLICK", "value": "", "position": [100, 200]}


def test_rightclick_literal_digits():
    text = "pyautogui.rightClick(50, 60)"
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "RIGHT_CLICK", "value": "", "position": [50, 60]}


def test_doubleclick_literal_digits():
    text = "pyautogui.doubleClick(300, 400)"
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "DOUBLE_CLICK", "value": "", "position": [300, 400]}


# --- variable-mediated coordinates (the ace-bench regression case) ---------------

def test_variable_mediated_moveto_maps_to_click():
    """This is the exact shape from writer-0810415c step 1 that produced 0%."""
    text = (
        "```python\n"
        "start_x, start_y = 651, 395\n"
        "end_x, end_y = 741, 524\n"
        "pyautogui.moveTo(start_x, start_y)\n"
        "```"
    )
    out = _agent()._try_parse_text_action(text)
    # Without mouseDown/mouseUp/dragTo, a bare moveTo(var, var) is a click.
    assert out == {"action": "CLICK", "value": "", "position": [651, 395]}


def test_variable_mediated_scalar_assignments():
    text = (
        "x = 42\n"
        "y = 84\n"
        "pyautogui.click(x, y)\n"
    )
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "CLICK", "value": "", "position": [42, 84]}


# --- drag detection ------------------------------------------------------------

def test_drag_via_dragTo_after_moveTo():
    text = (
        "pyautogui.moveTo(100, 200)\n"
        "pyautogui.dragTo(300, 400, duration=0.5)\n"
    )
    out = _agent()._try_parse_text_action(text)
    assert out["action"] == "DRAG"
    assert out["position"] == [300, 400]
    assert out["value"] == [100, 200]  # drag_from


def test_drag_via_moveDown_moveUp_sandwich():
    text = (
        "start_x, start_y = 651, 395\n"
        "end_x, end_y = 741, 524\n"
        "pyautogui.moveTo(start_x, start_y)\n"
        "pyautogui.mouseDown(button='left')\n"
        "pyautogui.moveTo(end_x, end_y)\n"
        "pyautogui.mouseUp(button='left')\n"
    )
    out = _agent()._try_parse_text_action(text)
    assert out["action"] == "DRAG"
    assert out["position"] == [741, 524]
    assert out["value"] == [651, 395]


# --- keyboard actions ----------------------------------------------------------

def test_typewrite_literal_string():
    text = "pyautogui.typewrite('hello world')"
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "INPUT", "value": "hello world", "position": [0, 0]}


def test_write_double_quotes():
    text = 'pyautogui.write("foo")'
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "INPUT", "value": "foo", "position": [0, 0]}


def test_press_key():
    text = "pyautogui.press('enter')"
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "KEY", "value": "enter", "position": [0, 0]}


def test_hotkey():
    text = "pyautogui.hotkey('ctrl', 's')"
    out = _agent()._try_parse_text_action(text)
    assert out["action"] == "HOTKEY"
    assert out["value"] == ["ctrl", "s"]


def test_scroll_negative():
    text = "pyautogui.scroll(-3)"
    out = _agent()._try_parse_text_action(text)
    assert out == {
        "action": "SCROLL",
        "value": -3,
        "position": [0, 0],
    }


# --- negative cases: pure prose should still parse via legacy regexes ----------

def test_at_coordinates_prose_still_works():
    text = "Click on the button at coordinates (400, 300) to submit."
    out = _agent()._try_parse_text_action(text)
    assert out == {"action": "CLICK", "value": "", "position": [400, 300]}


def test_no_actionable_content_returns_none():
    text = "I need to think about this task before proceeding."
    out = _agent()._try_parse_text_action(text)
    assert out is None


def test_empty_string_returns_none():
    assert _agent()._try_parse_text_action("") is None
    assert _agent()._try_parse_text_action(None) is None


def test_unknown_variable_falls_through_gracefully():
    """If a moveTo references an undefined variable, we shouldn't crash — we
    should either fall through to the legacy prose regex or return None."""
    text = "pyautogui.moveTo(unknown_var, other_var)"
    out = _agent()._try_parse_text_action(text)
    # Legacy regex won't find a bare digit tuple either → None.
    assert out is None


# --- _parse_response wiring + degraded-response logging ------------------------
#
# When gpugeek/Vendor2 silently drops tool_choice we should:
#   1. Log a shape inventory (stop_reason + block_types + foreign_tools) even
#      when a fallback rescues the response — for post-mortem attribution.
#   2. Route the TextBlock through _parse_pyautogui_block so the rescue lands.
#   3. When even the fallbacks fail, dump a bounded text_previews slice so we
#      can extend the parser to whatever new shape the upstream just emitted.


class _StubBlock:
    def __init__(self, type_, text=None, name=None, input_=None):
        self.type = type_
        if text is not None:
            self.text = text
        if name is not None:
            self.name = name
        if input_ is not None:
            self.input = input_


class _StubResp:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason


class _StubLoggerRecorder:
    """Captures logger.warning / logger.error messages verbatim."""

    def __init__(self) -> None:
        self.msgs: list[str] = []

    def _fmt(self, msg, args):
        try:
            return msg % args if args else msg
        except TypeError:
            return f"{msg} args={args!r}"

    def warning(self, msg, *args):
        self.msgs.append(self._fmt(msg, args))

    def error(self, msg, *args):  # pragma: no cover - not exercised here
        self.msgs.append(self._fmt(msg, args))

    def info(self, msg, *args):  # pragma: no cover
        pass


class _StubLoggerOwner:
    def __init__(self):
        self.logger = _StubLoggerRecorder()


_VAR_DRAG_FIXTURE = (
    "```python\n"
    "import pyautogui\n"
    "import time\n"
    "start_x, start_y = 651, 395\n"
    "end_x, end_y = 741, 524\n"
    "pyautogui.moveTo(start_x, start_y)\n"
    "pyautogui.mouseDown(button='left')\n"
    "pyautogui.moveTo(end_x, end_y, duration=0.8)\n"
    "pyautogui.mouseUp(button='left')\n"
    "```\n"
)


def test_parse_response_textblock_routes_to_drag():
    a = _agent()
    a.logger = _StubLoggerOwner()
    resp = _StubResp([_StubBlock("text", text=_VAR_DRAG_FIXTURE)])
    out = a._parse_response(resp)
    assert out["action"] == "DRAG"
    assert out["value"] == [651, 395]
    assert out["position"] == [741, 524]


def test_parse_response_logs_shape_inventory_on_degraded():
    a = _agent()
    a.logger = _StubLoggerOwner()
    resp = _StubResp([_StubBlock("text", text=_VAR_DRAG_FIXTURE)])
    a._parse_response(resp)
    inventory = [m for m in a.logger.logger.msgs if "degraded response" in m]
    assert inventory, f"no shape-inventory warning in {a.logger.logger.msgs!r}"
    # The inventory must name the block types we actually saw.
    assert "'text'" in inventory[0]


def test_parse_response_logs_text_previews_when_unparseable():
    a = _agent()
    a.logger = _StubLoggerOwner()
    resp = _StubResp([_StubBlock("text", text="I refuse to comply.")])
    out = a._parse_response(resp)
    assert out["action"] == "STOP"  # stop_reason=end_turn -> STOP
    previews = [m for m in a.logger.logger.msgs if "text_previews" in m]
    assert previews, f"no text_previews dump in {a.logger.logger.msgs!r}"
    assert "refuse to comply" in previews[0]


def test_parse_response_continue_when_not_end_turn():
    """If the model wasn't done talking, prefer CONTINUE over STOP so the
    planner gets another chance instead of the ace_agent conversion to DONE."""
    a = _agent()
    a.logger = _StubLoggerOwner()
    resp = _StubResp([_StubBlock("text", text="uh")], stop_reason="max_tokens")
    out = a._parse_response(resp)
    assert out["action"] == "CONTINUE"
