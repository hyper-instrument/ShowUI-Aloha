import os
import json
from typing import Dict, List

from ui_aloha.act.gui_agent.actor.ui_aloha_actor import AlohaActor
from ui_aloha.act.gui_agent.planner.ui_aloha_planner import AlohaPlanner
from ui_aloha.act.gui_agent.planner.trajectory_manager import TrajectoryManager
from ui_aloha.act.utils.visualize_utils import plot_action_vis
from ui_aloha.act.utils.app_utils import save_screenshot

def ui_aloha_loop(
    trajectory_manager: TrajectoryManager,
    planner: AlohaPlanner,
    actor: AlohaActor,
    task_id: str,
    query: str,
    screenshot: str,
    action_history: List[Dict] | List[str],
    trace_name: str = "default_trace",
    mode: str | None = None,
    log_dir: str = "./logs",
) -> Dict:
    """Run one iteration of the Aloha loop (plan → act).

    Args:
        trajectory_manager: Provides teach-mode in-context trajectory.
        planner: The planner component that produces plan JSON.
        actor: The actor component that produces an action.
        task_id: Task/session identifier.
        query: Natural language instruction.
        screenshot: Base64-encoded screenshot string.
        action_history: Prior actions or messages for context.
        trace_name: Named trajectory for teach-mode examples.
        mode: Per-request override of the actor backend (e.g.,
            "claude-computer-use"). When None, falls back to
            `actor.model` configured from `config.yaml: actor_model`.
        log_dir: Output directory for logs.

    Returns:
        Dict containing action, plan details, current step, and completion flag.
    """

    # Save screenshot
    screenshot_path = save_screenshot(screenshot, log_dir)

    # Resolve actor backend. Precedence:
    #   1. per-request `mode` (e.g. from HTTP payload)
    #   2. `actor.model` (from config.yaml: actor_model)
    #   3. hard fallback to "oai-operator"
    _SUPPORTED = {
        "oai-operator",
        "claude-computer-use",
        "claude-aloha-computer-use",
        "vanilla-claude",
        "ui-tars",
    }
    incoming_mode = (mode or "").lower().strip()
    configured_mode = (getattr(actor, "model", "") or "").lower().strip()
    if incoming_mode in _SUPPORTED:
        actor_mode = incoming_mode
    elif configured_mode in _SUPPORTED:
        actor_mode = configured_mode
    else:
        actor_mode = "oai-operator"

    # Vanilla mode: bypass the trajectory manager AND the planner. The actor
    # is fed the raw user task plus accumulated action_history and decides
    # the next action directly from the current screenshot. We still emit a
    # plan_details payload so the client/visualizer code paths don't break.
    if actor_mode == "vanilla-claude":
        action, complete_flag = actor(
            mode=actor_mode,
            messages=query,
            screenshot_path=screenshot_path,
            logging_dir=log_dir,
            action_history=action_history,
        )

        action_path = os.path.join(log_dir, f"actor_{actor_mode}.json")
        with open(action_path, "w") as f:
            json.dump(action, f, ensure_ascii=False, indent=4)

        action_vis_path = os.path.join(
            log_dir, f"actor_{actor_mode}_visualization.png"
        )
        plot_action_vis(action, screenshot_path, action_vis_path)

        plan_details = {
            "step_info": "vanilla mode: no planner / no trajectory",
            "observation": "",
            "reasoning": "",
            "action": "",
        }

        return {
            "action": action,
            "plan_details": plan_details,
            "curr_traj_step": 0,
            "complete_flag": complete_flag,
        }

    # Get guidance trajectory (teach-mode in-context)
    guidance_trajectory = trajectory_manager.get_trajectory_in_context(
        trace_name,
        formatting_string=True
    )
    
    # Save planning to log folder
    planning_path = os.path.join(log_dir, "planning_guidance_trajectory.json")
    with open(planning_path, "w") as f:
        json.dump(guidance_trajectory, f, ensure_ascii=False, indent=4)

    # Generate plan using AlohaPlanner
    planning = planner(
        task=query,
        guidance_trajectory=guidance_trajectory,
        screenshot_path=screenshot_path,
        action_history=action_history,
        logging_dir=log_dir,
    )

    planning_path = os.path.join(log_dir, "planning.json")
    with open(planning_path, "w") as f:
        json.dump(planning, f, ensure_ascii=False, indent=4)
    
    # Extract planner output fields
    planning_observation = planning.get('Observation', '')
    planning_next_action = planning.get('Action', '')
    planning_reasoning = planning.get('Reasoning', '')
    curr_traj_step = planning.get('Current Step', 1)
    curr_traj_step_explanation = planning.get('Current Step Explanation', '')

    # Generate action using Actor
    action, complete_flag = actor(
        mode=actor_mode,
        messages=planning,
        screenshot_path=screenshot_path,
        logging_dir=log_dir,
    )
    
    # Save action to log folder
    action_path = os.path.join(log_dir, f"actor_{actor_mode}.json")
    with open(action_path, "w") as f:
        json.dump(action, f, ensure_ascii=False, indent=4)

    # Draw action coord on screenshot
    action_vis_path = os.path.join(log_dir, f"actor_{actor_mode}_visualization.png")
    plot_action_vis(action, screenshot_path, action_vis_path)


    # Provide plan details for client visualization
    plan_details = {
        "step_info": curr_traj_step_explanation,
        "observation": planning_observation,
        "reasoning": planning_reasoning,
        "action": planning_next_action
    }

    # Return a dictionary with all the output values
    return {
        "action": action,
        "plan_details": plan_details,
        "curr_traj_step": curr_traj_step,
        "complete_flag": complete_flag,
    }
