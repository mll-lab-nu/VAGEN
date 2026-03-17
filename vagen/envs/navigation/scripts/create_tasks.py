"""Generate navigation training datasets for AI2-THOR environments.

Produces JSON files containing navigation tasks with randomized agent
start positions and target objects across AI2-THOR scenes.

Usage:
    # Generate base training set (direct instructions):
    python create_tasks.py --dataset_type base --output ../datasets/base_train.json

    # Generate common-sense training set (indirect instructions):
    python create_tasks.py --dataset_type common_sense --output ../datasets/common_sense_train.json

    # Append more tasks to an existing file:
    python create_tasks.py --dataset_type base --output ../datasets/base_train.json --append
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import ai2thor.controller
import numpy as np
from ai2thor.platform import CloudRendering

# AI2-THOR scene ranges: kitchens (1-10), living rooms (201-210),
# bedrooms (301-310), bathrooms (401-410)
SCENE_RANGES = [range(1, 11), range(201, 211), range(301, 311), range(401, 411)]
ALL_SCENES = [f"FloorPlan{i}" for r in SCENE_RANGES for i in r]

# Objects considered valid navigation targets
INTERACTIVE_OBJECT_TYPES = {
    "StoveBurner", "Microwave", "Sink", "Bathtub", "Toilet", "Laptop", "TV",
}

DEFAULT_TASKS_PER_SCENE = 20
MAX_RETRIES_PER_TASK = 200
MIN_AGENT_TARGET_DIST = 1.0
GRID_SIZE = 0.1  


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate navigation training tasks.")
    parser.add_argument("--output", type=str, default="base_train.json",
                        help="Output JSON file path")
    parser.add_argument("--tasks_per_scene", type=int, default=DEFAULT_TASKS_PER_SCENE,
                        help="Number of tasks to generate per scene")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing file instead of overwriting")
    parser.add_argument("--dataset_type", type=str, choices=["base", "common_sense"],
                        default="base", help="Type of instructions to generate")
    return parser.parse_args()


def load_existing_tasks(filename: str) -> List[Dict[str, Any]]:
    """Load existing tasks from a file to support the --append flag."""
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, "r") as f:
            return json.load(f).get("tasks", [])
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {filename}. Starting fresh. Error: {e}")
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
    from common_sense_templates import OBJECT_DESCRIPTIONS, SUFFIXES

    suffix = random.choice(SUFFIXES)
    descriptions = OBJECT_DESCRIPTIONS.get(object_type)
    if descriptions:
        return f"{random.choice(descriptions)} {suffix}"
    return f"I am looking for the {object_type} to complete a task. {suffix}"


def generate() -> None:
    args = parse_args()

    current_tasks: List[Dict[str, Any]] = []
    if args.append:
        current_tasks = load_existing_tasks(args.output)
        print(f"Loaded {len(current_tasks)} existing tasks. Appending...")

    print("Initializing AI2-THOR...")
    controller = ai2thor.controller.Controller(
        agentMode="default",
        visibilityDistance=1.5,
        gridSize=GRID_SIZE,
        scene="FloorPlan1",
        width=300,
        height=300,
        platform=CloudRendering,
    )

    for scene in ALL_SCENES:
        try:
            controller.reset(scene=scene)
        except Exception:
            continue

        event = controller.step(action="Pass")
        all_objects = {o["objectId"]: o for o in event.metadata["objects"]}

        valid_targets = [
            obj for obj in event.metadata["objects"]
            if (obj["pickupable"] or obj["objectType"] in INTERACTIVE_OBJECT_TYPES)
            and not is_object_hidden(obj, all_objects)
        ]

        event = controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]

        if not valid_targets or not reachable_positions:
            continue

        scene_count = 0
        retries = 0

        while scene_count < args.tasks_per_scene and retries < (args.tasks_per_scene * MAX_RETRIES_PER_TASK):
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

        print(f"  {scene}: Generated {scene_count} tasks.")

    controller.stop()

    with open(args.output, "w") as f:
        json.dump({"tasks": current_tasks}, f, indent=4)
    print(f"Saved {len(current_tasks)} tasks to {args.output}")


if __name__ == "__main__":
    generate()
