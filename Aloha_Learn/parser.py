# parser.py
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass

import os
import glob
from pathlib import Path

from log_processor import LogProcessor
from screenshot_processor import VideoScreenshotExtractor
from trace_generator import TraceGenerator


def _resolve_project_dir(project_name: str) -> Path:
    """
    Accept either a bare name ('Drag_0') or a full path.
    If bare name, resolve to ./projects/{project_name}.
    """
    p = Path(project_name)
    if p.exists():
        return p.resolve()
    cand = Path.cwd() / "projects" / project_name
    if cand.exists():
        return cand.resolve()
    raise FileNotFoundError(f"Project folder not found. Tried: {p} and {cand}")


def _find_single_log(inputs_dir: Path) -> Path:
    """
    Find exactly one log file in inputs_dir with extensions .txt, .log, .json.
    Match the behavior expected by the existing LogProcessor CLI.
    """
    hits = []
    for ext in ("*.txt", "*.log", "*.json"):
        hits.extend(inputs_dir.glob(ext))
    if not hits:
        raise FileNotFoundError(f"No log files in {inputs_dir} (accepted: .txt, .log, .json)")
    if len(hits) > 1:
        names = ", ".join(h.name for h in hits)
        raise RuntimeError(f"Multiple log files in {inputs_dir}: {names}. Keep only one.")
    return hits[0]


def _build_meta_json(project_dir: Path, raw_log: Path, processed_log: Path,
                       log_sc: Path, trace_path: Path, meta: dict,
                       overall_task: str) -> dict:
    """Aggregate pipeline artefacts into a machine-readable meta.json."""
    import json as _json

    # Load processed log for action stats
    actions = []
    if processed_log.exists():
        with open(processed_log, "r", encoding="utf-8") as f:
            actions = _json.load(f)

    # Load trace for step count and overall_task fallback
    trace = {}
    if trace_path.exists():
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = _json.load(f)

    # Extract screen info from CONFIG action
    screen_info = {}
    for act in actions:
        if act.get("action") == "CONFIG" and isinstance(act.get("coords"), dict):
            screen_info = act["coords"]
            break

    # Software list from log
    softwares = sorted({a.get("current_software") for a in actions if a.get("current_software")})

    # Action breakdown
    from collections import Counter
    breakdown = Counter()
    for a in actions:
        raw = a.get("action", "")
        if raw.startswith("LClick"):
            breakdown["LClick"] += 1
        elif raw.startswith("RClick"):
            breakdown["RClick"] += 1
        elif raw.startswith("Key Press"):
            breakdown["KeyPress"] += 1
        elif raw.startswith("Key Release"):
            breakdown["KeyRelease"] += 1
        elif raw.startswith("Hotkey"):
            breakdown["Hotkey"] += 1
        elif raw.startswith("Scroll"):
            breakdown["Scroll"] += 1
        elif raw == "CONFIG":
            breakdown["CONFIG"] += 1
        else:
            breakdown[raw] += 1

    # Duration
    duration = 0.0
    if len(actions) >= 2:
        duration = actions[-1].get("timestamp", 0.0) - actions[0].get("timestamp", 0.0)

    # Screenshot counts
    screenshots_dir = project_dir / "screenshots"
    num_full = len(list(screenshots_dir.glob("*.jpg"))) if screenshots_dir.exists() else 0
    num_crop = len([p for p in screenshots_dir.glob("*.jpg") if ".crop." in p.name]) if screenshots_dir.exists() else 0

    meta_doc = {
        "aloha_version": "ShowUI-Aloha",
        "task": {
            "name": project_dir.name,
            "overall_task": overall_task or trace.get("overall_task", ""),
            "description": "",
        },
        "instruments": {
            "recording_software": "Aloha Screen Recorder",
            "target_applications": [s for s in softwares if "Screen Recorder" not in s and s != "System Info"],
            "operating_system": "macOS" if any("Screen Recorder" in s for s in softwares) else "unknown",
            "screen_info": screen_info,
        },
        "recording": {
            "raw_log_file": raw_log.name,
            "processed_log_file": processed_log.name,
            "screenshots_dir": str(screenshots_dir.relative_to(project_dir)) if screenshots_dir.exists() else "screenshots",
            "trace_file": trace_path.name,
            "duration_seconds": round(duration, 3),
            "num_actions": len(actions),
        },
        "pipeline": {
            "coordinate_scaling": meta.get("coordinate_scaling", False),
            "original_resolution": meta.get("original_resolution", "unknown"),
            "target_resolution": meta.get("target_resolution", "unknown"),
        },
        "data": {
            "num_full_screenshots": num_full - num_crop,
            "num_crop_screenshots": num_crop,
            "num_trace_steps": len(trace.get("trajectory", [])),
            "action_breakdown": dict(breakdown),
        },
    }
    return meta_doc


def run_pipeline(project_name: str, overall_task: str = "") -> Path:
    """
    Orchestrate the 3-step pipeline:
      1) parse & merge events -> {project}_processed_log.json
      2) extract screenshots + crops -> {project}_processed_log_sc.json
      3) LLM trace generation -> {project}_trace.json
      4) meta.json -> aggregated dataset metadata

    Args:
        project_name: Project folder name or path under Aloha_Learn conventions.
        overall_task: Optional natural-language description of the whole recording; passed into
            each caption-generation prompt so the LLM interprets clicks/keys in context.
            Note: screenshot crop and coordinate scaling are deterministic (OpenCV); this hint
            does not change that stage unless you extend screenshot_processor/log_processor.

    Returns final trace path.
    """
    project_dir = _resolve_project_dir(project_name)
    inputs_dir = project_dir / "inputs"
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Inputs directory not found: {inputs_dir}")

    # ---------- Step 1: process raw log -> processed log ----------
    raw_log = _find_single_log(inputs_dir)
    processed_log_path = project_dir / f"{project_dir.name}_processed_log.json"

    lp = LogProcessor()
    # keep default typing-delay behavior (5.0s) to align with existing logic
    lp.process_log_file(str(raw_log), str(processed_log_path), time_threshold=5.0)

    # ---------- Step 2: screenshots + scaled coords -> *_processed_log_sc.json ----------
    vse = VideoScreenshotExtractor()
    # This function expects the processed log with the exact filename in the project root.
    # It will discover the video and create {project}_processed_log_sc.json and /screenshots.
    _, screenshots_dir, meta = vse.process_project(str(project_dir))

    # ---------- Step 3: generate LLM trace -> {project}_trace.json ----------
    log_sc = project_dir / f"{project_dir.name}_processed_log_sc.json"
    if not log_sc.exists():
        raise FileNotFoundError(f"Expected processed-with-screenshots log not found: {log_sc}")

    out_trace = project_dir / f"{project_dir.name}_trace.json"

    tg = TraceGenerator(
        default_prompt_path="default_prompt.json",
        api_provider="claude",
        openai_model="gpt-4o",
        claude_model="claude-sonnet-4-20250514",
        api_keys_path="config/api_keys.json",
    )
    tg.generate_trace(
        recording_json_path=str(log_sc),
        screenshots_dir=str(screenshots_dir),
        output_trace_path=str(out_trace),
        overall_task=overall_task,
    )

    # ---------- Step 4: meta.json ----------
    meta_doc = _build_meta_json(
        project_dir=project_dir,
        raw_log=raw_log,
        processed_log=processed_log_path,
        log_sc=log_sc,
        trace_path=out_trace,
        meta=meta,
        overall_task=overall_task,
    )
    meta_path = project_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, ensure_ascii=False, indent=2)

    print("=== Pipeline Complete ===")
    print(f"Project: {project_dir.name}")
    print(f"Processed log: {processed_log_path.name}")
    print(f"Screenshots dir: {Path(screenshots_dir).name}")
    print(f"Processed log (+screens): {log_sc.name}")
    print(f"Trace: {out_trace.name}")
    print(f"Meta: {meta_path.name}")
    return out_trace


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full Parser pipeline (1) log → (2) screenshots → (3) trace")
    parser.add_argument("project_name", help="Either a bare name (e.g., 'Drag_0') or a full path to the project folder.")
    parser.add_argument(
        "--task",
        "-t",
        default="",
        help="Whole-video task hint for trace captions (passed into LLM as Overall Task).",
    )
    parser.add_argument(
        "--task-file",
        default=None,
        metavar="PATH",
        help="UTF-8 file whose contents replace --task (for long prompts).",
    )
    args = parser.parse_args()
    task_hint = args.task or ""
    if args.task_file:
        tf = Path(args.task_file).expanduser()
        task_hint = tf.read_text(encoding="utf-8").strip()
    run_pipeline(args.project_name, overall_task=task_hint)
