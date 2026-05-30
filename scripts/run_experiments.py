from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))

from ultralytics import YOLO  # noqa: E402
from yolo_custom import register_custom_modules  # noqa: E402

register_custom_modules()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_args(config: dict[str, Any], experiment: dict[str, Any], group: str) -> dict[str, Any]:
    defaults = dict(config.get("defaults", {}))
    defaults.update({k: v for k, v in experiment.items() if k not in {"name", "model"}})
    defaults.update(
        {
            "data": config["data"],
            "project": config.get("project", "runs/paper"),
            "name": f"{group}_{experiment['name']}",
            "device": config.get("device", 0),
            "seed": config.get("seed", 42),
            "exist_ok": True,
        }
    )
    return defaults


def run_group(config: dict[str, Any], group: str, dry_run: bool = False) -> None:
    experiments = config.get(group, [])
    if not experiments:
        raise ValueError(f"No experiments found for group: {group}")

    for exp in experiments:
        train_args = build_args(config, exp, group)
        print(f"\n[{group}] {exp['name']}")
        print(f"model: {exp['model']}")
        print("args:", train_args)
        if dry_run:
            continue
        model = YOLO(exp["model"])
        model.train(**train_args)
        model.val(data=config["data"], project=config.get("project", "runs/paper"), name=f"{group}_{exp['name']}_val", device=config.get("device", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO comparison and ablation experiments.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Path to experiment YAML.")
    parser.add_argument("--group", choices=["comparison", "ablation", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    groups = ["comparison", "ablation"] if args.group == "all" else [args.group]
    for group in groups:
        run_group(config, group, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
