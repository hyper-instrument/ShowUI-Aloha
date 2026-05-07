import os
import json
import logging
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

class TrajectoryManager:
    """
    Manages user trajectory data for task execution recordings.
    Provides methods to load, access, and format trajectory information.
    """
    def __init__(self, base_path: str = r"./cache"):

        self.base_path = base_path
        
        
    def get_full_trace(self, trace_name: str) -> Optional[Dict]:
        """
        Load trace data for a specific trace.

        Resolution order (first hit wins):
          1) base_path/{trace_name}_trace.json   (Aloha_Learn parser output)
          2) base_path/{trace_name}.json         (legacy / hand-written)
          3) base_path/{trace_name}              (raw filename, no extension)
          4) base_path/{trace_name}/trace.json   (per-trace folder layout)
        """
        # Strip any extension/suffix the caller might have passed in.
        clean = trace_name
        for suffix in ("_trace.json", ".json"):
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
                break

        candidate_paths = [
            os.path.join(self.base_path, f"{clean}_trace.json"),
            os.path.join(self.base_path, f"{clean}.json"),
            os.path.join(self.base_path, clean),
            os.path.join(self.base_path, clean, "trace.json"),
        ]

        file_path = None
        for path in candidate_paths:
            if os.path.isfile(path):
                file_path = path
                break

        if file_path is None:
            log.warning(
                "Trace file not found for %r. Tried: %s",
                trace_name,
                ", ".join(candidate_paths),
            )
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            log.warning("JSON parsing error in %s: %s", file_path, e)
            return None
        except OSError as e:
            log.warning("Could not read trace %s: %s", file_path, e)
            return None


    def get_trajectory_in_context(self, trace_name: str, formatting_string: bool = True) -> Optional[str]:
        """
        Get the in-context example for the given trace.
        
        Args:
            trace_name (str): Name of the trace
            formatting_string (bool): Whether to format the output as string (True) or list (False)
            
        Returns:
            Optional[str]: Formatted in-context example string/list or None if trace not found
        """
        
        trace_data = self.get_full_trace(trace_name)
        if not trace_data:
            return None
        
        steps = trace_data.get("trajectory", [])
        context_steps = []

        overall = trace_data.get("overall_task")
        if overall is not None and str(overall).strip():
            context_steps.append(f"Overall goal (recording): {str(overall).strip()}")

        for action in steps:
            
            if "milestone" in action:  # filter out 'milestones'
                continue
            
            step_idx = action['step_idx']
            step_caption = action['caption']
            step_action = step_caption['action']
            context_steps.append(f"Step [{step_idx}]: {step_action}")
        
        if formatting_string:
            return "\n".join(context_steps)
        else:
            return context_steps
