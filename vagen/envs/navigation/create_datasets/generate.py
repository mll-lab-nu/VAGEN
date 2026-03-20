"""Generate navigation training datasets for AI2-THOR environments.

Produces JSON files containing navigation tasks with randomized agent
start positions and target objects across AI2-THOR scenes.

Training scenes are chosen to NOT overlap with eval scenes:
  - Eval uses: FloorPlan11-30, 211-230, 311-330 (60 scenes)
  - Train uses: FloorPlan1-10, 201-210, 301-310, 401-430 (60 scenes)

Usage:
    python -m vagen.envs.navigation.create_datasets.generate \
        --dataset_type base --output base_train.json

    python -m vagen.envs.navigation.create_datasets.generate \
        --dataset_type common_sense --output common_sense_train.json

    python -m vagen.envs.navigation.create_datasets.generate \
        --dataset_type base --tasks_per_scene 30 --output base_train.json

    python -m vagen.envs.navigation.create_datasets.generate \
        --dataset_type long_horizon --output long_horizon_train.json

    python -m vagen.envs.navigation.create_datasets.generate \
        --dataset_type base --output base_train.json --append
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List

import numpy as np

from .common_sense_templates import OBJECT_DESCRIPTIONS, SUFFIXES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AI2-THOR scene definitions
# ---------------------------------------------------------------------------
# Full AI2-THOR scene ranges per room type (30 each):
#   Kitchens:     FloorPlan1  - FloorPlan30
#   Living rooms: FloorPlan201 - FloorPlan230
#   Bedrooms:     FloorPlan301 - FloorPlan330
#   Bathrooms:    FloorPlan401 - FloorPlan430
#
# Eval scenes (from EmbodiedBench):
#   FloorPlan11-30, FloorPlan211-230, FloorPlan311-330
#
# Training scenes (no overlap with eval):
TRAIN_SCENE_RANGES = [
    range(1, 11),      # Kitchens 1-10
    range(201, 211),    # Living rooms 201-210
    range(301, 311),    # Bedrooms 301-310
    range(401, 431),    # Bathrooms 401-430  (eval doesn't use bathrooms)
]
TRAIN_SCENES = [f"FloorPlan{i}" for r in TRAIN_SCENE_RANGES for i in r]

# Sanity check: no overlap with eval
EVAL_SCENE_RANGES = [range(11, 31), range(211, 231), range(311, 331)]
_EVAL_SCENES = {f"FloorPlan{i}" for r in EVAL_SCENE_RANGES for i in r}
assert not (set(TRAIN_SCENES) & _EVAL_SCENES), "Train/eval scene overlap detected!"

# Objects considered valid navigation targets (non-pickupable but interactive)
INTERACTIVE_OBJECT_TYPES = {
    "StoveBurner", "Microwave", "Fridge", "Toaster", "Sink",
    "Bathtub", "Toilet", "Laptop", "Television", "CoffeeMachine",
    "GarbageCan", "DeskLamp", "FloorLamp", "Safe", "LaundryHamper",
}

DEFAULT_TASKS_PER_SCENE = 20
MAX_RETRIES_PER_TASK = 200
MIN_AGENT_TARGET_DIST = 2.0   # Raised from PR's 1.0 to avoid trivial tasks
GRID_SIZE = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate navigation training tasks.")
    parser.add_argument("--output", type=str, default="base_train.json",
                        help="Output JSON file name (saved to ../assets/)")
    parser.add_argument("--tasks_per_scene", type=int, default=DEFAULT_TASKS_PER_SCENE,
                        help="Number of tasks to generate per scene")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing file instead of overwriting")
    parser.add_argument("--dataset_type", type=str, choices=["base", "common_sense", "long_horizon"],
                        default="base", help="Type of instructions to generate")
    parser.add_argument("--gpu_device", type=int, default=0,
                        help="GPU device ID for AI2-THOR rendering")
    return parser.parse_args()


def load_existing_tasks(filename: str) -> List[Dict[str, Any]]:
    """Load existing tasks from a file to support the --append flag."""
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r") as f:
            return json.load(f).get("tasks", [])
    except (json.JSONDecodeError, IOError) as e:
        LOGGER.warning("Could not load %s. Starting fresh. Error: %s", filename, e)
        return []


def is_object_hidden(obj: Dict[str, Any], all_objects: Dict[str, Dict[str, Any]]) -> bool:
    """Check if an object is inside a closed receptacle (not visible to agent)."""
    if not obj.get("parentReceptacles"):
        return False
    for parent_id in obj["parentReceptacles"]:
        parent = all_objects.get(parent_id)
        if parent and parent.get("openable") and not parent.get("isOpen"):
            return True
    return False


def get_base_instruction(object_type: str) -> str:
    """Generate a direct navigation instruction."""
    return f"navigate to the {object_type} in the room and be as close as possible to it"


def get_common_sense_instruction(object_type: str) -> str:
    """Generate an indirect, common-sense-style navigation instruction."""
    suffix = random.choice(SUFFIXES)
    descriptions = OBJECT_DESCRIPTIONS.get(object_type)
    if descriptions:
        return f"{random.choice(descriptions)} {suffix}"
    return f"I am looking for the {object_type} to complete a task. {suffix}"


def get_away_facing_rotation(agent_pos: Dict[str, float], target_pos: Dict[str, float]) -> float:
    """Return a rotation (0/90/180/270) that faces AWAY from the target.

    Computes the angle from agent to target, then picks the closest
    cardinal direction that points in the opposite direction.
    AI2-THOR rotation convention: 0=+Z, 90=+X, 180=-Z, 270=-X.
    """
    dx = target_pos["x"] - agent_pos["x"]
    dz = target_pos["z"] - agent_pos["z"]
    # Angle toward target in degrees (AI2-THOR: 0=+Z, clockwise)
    toward_angle = np.degrees(np.arctan2(dx, dz)) % 360
    # Opposite direction
    away_angle = (toward_angle + 180) % 360
    # Snap to nearest cardinal direction
    cardinals = [0.0, 90.0, 180.0, 270.0]
    return min(cardinals, key=lambda c: min(abs(away_angle - c), 360 - abs(away_angle - c)))


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate() -> None:
    args = parse_args()

    # Output goes to ../assets/ relative to this package
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
    output_path = os.path.join(datasets_dir, args.output) if not os.path.isabs(args.output) else args.output

    current_tasks: List[Dict[str, Any]] = []
    if args.append:
        current_tasks = load_existing_tasks(output_path)
        LOGGER.info("Loaded %d existing tasks. Appending...", len(current_tasks))

    LOGGER.info("Initializing AI2-THOR (GPU %d)...", args.gpu_device)
    LOGGER.info("Training scenes: %d total (%s ... %s)",
                len(TRAIN_SCENES), TRAIN_SCENES[0], TRAIN_SCENES[-1])

    import ai2thor.controller
    from ai2thor.platform import CloudRendering

    controller = ai2thor.controller.Controller(
        agentMode="default",
        visibilityDistance=10,
        gridSize=GRID_SIZE,
        scene="FloorPlan1",
        width=300,
        height=300,
        fieldOfView=100,
        platform=CloudRendering,
        gpu_device=args.gpu_device,
        server_timeout=300,
        server_start_timeout=300,
    )

    total_generated = 0
    failed_scenes: List[str] = []

    try:
        for scene_idx, scene in enumerate(TRAIN_SCENES):
            try:
                controller.reset(scene=scene)
            except Exception as e:
                LOGGER.warning("Failed to load scene %s: %s", scene, e)
                failed_scenes.append(scene)
                continue

            event = controller.step(action="Pass")
            all_objects = {o["objectId"]: o for o in event.metadata["objects"]}

            # Valid targets: pickupable objects OR interactive fixtures, not hidden
            valid_targets = [
                obj for obj in event.metadata["objects"]
                if (obj["pickupable"] or obj["objectType"] in INTERACTIVE_OBJECT_TYPES)
                and not is_object_hidden(obj, all_objects)
            ]

            event = controller.step(action="GetReachablePositions")
            reachable_positions = event.metadata["actionReturn"]

            if not valid_targets or not reachable_positions:
                LOGGER.warning("Scene %s: no valid targets or positions, skipping", scene)
                failed_scenes.append(scene)
                continue

            scene_count = 0
            retries = 0
            max_retries = args.tasks_per_scene * MAX_RETRIES_PER_TASK

            while scene_count < args.tasks_per_scene and retries < max_retries:
                retries += 1
                target = random.choice(valid_targets)
                start_pos = random.choice(reachable_positions)
                rotation = random.choice([0.0, 90.0, 180.0, 270.0])

                dist = np.sqrt(
                    (start_pos["x"] - target["position"]["x"]) ** 2
                    + (start_pos["z"] - target["position"]["z"]) ** 2
                )
                if dist < MIN_AGENT_TARGET_DIST:
                    continue

                if args.dataset_type == "common_sense":
                    instruction = get_common_sense_instruction(target["objectType"])
                else:
                    instruction = get_base_instruction(target["objectType"])

                # Long-horizon: face away from target for harder exploration
                if args.dataset_type == "long_horizon":
                    rotation = get_away_facing_rotation(start_pos, target["position"])

                task = {
                    "targetObjectType": target["objectType"],
                    "targetObjectIds": target["objectId"],
                    "target_position": target["position"],
                    "agentPose": {
                        "position": start_pos,
                        "rotation": rotation,
                        "horizon": 0.0,
                    },
                    "scene": scene,
                    "object_to_hide": [],
                    "instruction": instruction,
                }
                current_tasks.append(task)
                scene_count += 1

            total_generated += scene_count
            LOGGER.info("  [%d/%d] %s: %d tasks (retries: %d, objects: %d)",
                        scene_idx + 1, len(TRAIN_SCENES), scene, scene_count,
                        retries, len(valid_targets))

    finally:
        controller.stop()
        LOGGER.info("Controller stopped.")

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"tasks": current_tasks}, f, indent=4)

    LOGGER.info("Saved %d tasks to %s", len(current_tasks), output_path)
    LOGGER.info("Total generated this run: %d across %d/%d scenes",
                total_generated, len(TRAIN_SCENES) - len(failed_scenes), len(TRAIN_SCENES))
    if failed_scenes:
        LOGGER.warning("Failed scenes (%d): %s", len(failed_scenes), failed_scenes)


if __name__ == "__main__":
    generate()
