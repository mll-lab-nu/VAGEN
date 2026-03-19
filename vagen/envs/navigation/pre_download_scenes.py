"""Pre-download all AI2-THOR scenes used by the navigation environment.

Scans all *.json files under the adjacent assets/ folder to collect
every unique scene, then creates a Controller and resets each scene
once so AI2-THOR caches the binary + scene assets under ~/.ai2thor/.

Usage:
    conda activate viewsuite
    python -m vagen.envs.navigation.pre_download_scenes
"""

import argparse
import json
import glob
import os
import time

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def get_all_scenes():
    scenes = set()
    for fpath in sorted(glob.glob(os.path.join(DATASETS_DIR, "*.json"))):
        with open(fpath) as f:
            data = json.load(f)
        # support both {"tasks": [...]} and bare [...]
        tasks = data.get("tasks", data) if isinstance(data, dict) else data
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if isinstance(task, dict) and "scene" in task:
                scenes.add(task["scene"])
        print(f"  {os.path.basename(fpath)}: {len(tasks)} tasks")
    return sorted(scenes)


def main():
    parser = argparse.ArgumentParser(description="Pre-download AI2-THOR scene assets")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device ID (default: auto-detect first available)")
    args = parser.parse_args()

    # Auto-detect first available GPU if not specified
    gpu = args.gpu
    if gpu is None:
        import subprocess
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
            )
            gpu = int(out.strip().split("\n")[0])
        except Exception:
            gpu = 0
        print(f"Auto-detected GPU {gpu}")

    print(f"Scanning {DATASETS_DIR} ...")
    scenes = get_all_scenes()
    print(f"\nFound {len(scenes)} unique scenes to pre-load.\n")

    import ai2thor.controller
    from ai2thor.platform import CloudRendering

    print(f"Creating AI2-THOR controller on GPU {gpu} ...")
    controller = ai2thor.controller.Controller(
        agentMode="default",
        gridSize=0.1,
        visibilityDistance=10,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=255,
        height=255,
        fieldOfView=100,
        platform=CloudRendering,
        gpu_device=gpu,
        server_timeout=300,
        server_start_timeout=300,
    )
    print("Controller created.\n")

    t_start = time.time()
    try:
        for i, scene in enumerate(scenes, 1):
            t0 = time.time()
            print(f"[{i}/{len(scenes)}] {scene} ...", end=" ", flush=True)
            controller.reset(scene=scene)
            print(f"ok ({time.time() - t0:.1f}s)")
    finally:
        controller.stop()
    print(f"\nDone! {len(scenes)} scenes cached in {time.time() - t_start:.0f}s")
    print("Assets are stored in ~/.ai2thor/")


if __name__ == "__main__":
    main()
