from typing import Any

import argparse
import time
import threading
import platform
import os
import logging

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

from flask import Flask, request, jsonify

from ui_aloha.execute.executor.aloha_executor import AlohaExecutor
from ui_aloha.execute.sampling_loop import simple_sampling_loop


class SharedState:
    def __init__(self, args):
        self.args = args
        self.task = getattr(args, 'task', "")
        self.selected_screen = args.selected_screen
        self.trace_id = args.trace_id
        self.server_url = args.server_url
        self.max_steps = getattr(args, 'max_steps', 50)
        # Set per /run_task request (JSON `mode` or `actor_model`); not a CLI flag.
        self.mode: str | None = None

        self.is_processing = False
        self.should_stop = False
        self.stop_event = threading.Event()
        self.processing_thread: threading.Thread | None = None


shared_state: SharedState | None = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def process_input():
    global shared_state
    assert shared_state is not None
    logging.info("process_input thread started.")
    shared_state.is_processing = True
    shared_state.should_stop = False
    shared_state.stop_event.clear()

    try:
        sampling_loop = simple_sampling_loop(
            task=shared_state.task,
            selected_screen=shared_state.selected_screen,
            trace_id=shared_state.trace_id,
            server_url=shared_state.server_url,
            max_steps=shared_state.max_steps,
            mode=shared_state.mode,
        )

        for loop_msg in sampling_loop:
            if shared_state.should_stop or shared_state.stop_event.is_set():
                break

            # Progress logs: full text; avoid dumping megabytes of base64 screens.
            try:
                msg_type = loop_msg.get("type")
                raw = loop_msg.get("content")
                if msg_type == "image_base64" and isinstance(raw, str):
                    head = 120
                    snippet = raw[:head] + ("..." if len(raw) > head else "")
                    logging.info(
                        "[loop_msg] type=%s content_len=%s content_prefix=%s",
                        msg_type,
                        len(raw),
                        snippet,
                    )
                else:
                    logging.info("[loop_msg] type=%s content=%s", msg_type, raw)
            except Exception:
                logging.info("[loop_msg] %s", loop_msg)

            # light pacing to avoid busy loop in UI
            time.sleep(0.1)

            if shared_state.should_stop or shared_state.stop_event.is_set():
                break

    except Exception as e:
        logging.error(f"Error during task processing: {e}", exc_info=True)
    finally:
        shared_state.is_processing = False
        shared_state.should_stop = False
        shared_state.stop_event.clear()
        logging.info("process_input thread finished.")


@app.route("/run_task", methods=["POST"])
def run_task():
    """Start a background task that chats with the server and executes actions locally."""
    data = request.get_json(silent=True) or {}
    required = ["task"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"status": "error", "message": f"Missing required field(s): {', '.join(missing)}"}), 400

    assert shared_state is not None
    if shared_state.is_processing:
        return jsonify({"status": "error", "message": "A task is already running"}), 409

    # Update runtime parameters if provided
    shared_state.task = data.get("task", shared_state.task)
    shared_state.selected_screen = data.get("selected_screen", shared_state.selected_screen)
    shared_state.trace_id = data.get("trace_id", shared_state.trace_id)
    shared_state.server_url = data.get("server_url", shared_state.server_url)
    shared_state.max_steps = data.get("max_steps", shared_state.max_steps)
    # Accept either `mode` or `actor_mode` for the per-request actor override
    # (e.g. "vanilla-claude" to skip trace/planner). Empty string clears it.
    incoming_mode = data.get("mode", data.get("actor_mode", shared_state.mode))
    if isinstance(incoming_mode, str):
        incoming_mode = incoming_mode.strip() or None
    shared_state.mode = incoming_mode

    shared_state.stop_event.clear()
    shared_state.processing_thread = threading.Thread(target=process_input, daemon=True)
    shared_state.processing_thread.start()

    return jsonify({
        "status": "success",
        "message": "Task started",
        "task": shared_state.task,
        "mode": shared_state.mode,
    })


@app.route("/stop", methods=["POST"])
def stop():
    assert shared_state is not None
    if not shared_state.is_processing:
        return jsonify({"status": "error", "message": "No active task to stop"}), 400

    shared_state.should_stop = True
    shared_state.stop_event.set()

    return jsonify({"status": "success", "message": "Stop signal sent"})


def main():
    logging.info("App main() function starting setup.")
    global shared_state
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Following the instructions to complete the task.", help="Task description")
    parser.add_argument("--selected_screen", type=int, default=0, help="Selected screen index")
    parser.add_argument("--trace_id", type=str, default="example_trace", help="Trace ID for the session")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://127.0.0.1:7887/generate_action",
        help="Action server endpoint",
    )
    parser.add_argument("--max_steps", type=int, default=50)

    args = parser.parse_args()

    shared_state = SharedState(args)
    logging.info("Shared state initialized.")

    port = 7888
    host = "0.0.0.0"
    logging.info(f"Starting Client Flask on {host}:{port}")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
