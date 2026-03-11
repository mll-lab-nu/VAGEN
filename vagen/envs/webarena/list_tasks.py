"""List all WebArena tasks sorted by ID."""
import json
import glob
import os

DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(DIR, "config_files")


def main():
    tasks = []
    for fpath in glob.glob(os.path.join(CONFIG_DIR, "*.json")):
        fname = os.path.basename(fpath)
        if fname in ("test.json", "test.raw.json"):
            continue
        idx = int(fname.replace(".json", ""))
        with open(fpath) as f:
            data = json.load(f)
        tasks.append({
            "id": idx,
            "intent": data.get("intent", ""),
            "sites": data.get("sites", []),
            "eval_types": data.get("eval", {}).get("eval_types", []),
        })
    tasks.sort(key=lambda t: t["id"])

    print(f"{'ID':>4}  {'Sites':<25}  {'Eval Types':<35}  Intent")
    print("-" * 140)
    for t in tasks:
        sites = ",".join(t["sites"])
        eval_types = ",".join(t["eval_types"])
        print(f"{t['id']:>4}  {sites:<25}  {eval_types:<35}  {t['intent']}")
    print("-" * 140)
    print(f"\nTotal: {len(tasks)}")


if __name__ == "__main__":
    main()
