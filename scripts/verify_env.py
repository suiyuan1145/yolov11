from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))

import torch  # noqa: E402
import ultralytics  # noqa: E402


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"ultralytics: {ultralytics.__version__}")
    print(f"yolo_config_dir: {os.environ['YOLO_CONFIG_DIR']}")


if __name__ == "__main__":
    main()
