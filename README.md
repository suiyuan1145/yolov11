# YOLO Paper Experiments

This workspace is prepared for YOLO comparison experiments and ablation experiments.

## Environment

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify CUDA and Ultralytics:

```powershell
$env:YOLO_CONFIG_DIR = (Resolve-Path .yolo_config).Path
python scripts/verify_env.py
```

Current verified setup:

- Python 3.12.10
- PyTorch 2.11.0+cu128
- Ultralytics 8.4.56
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU

Linux setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
unzip archive.zip
```

## Dataset

The dataset from `archive.zip` should be extracted to:

```text
archive/
```

The active YOLO data config is:

```text
dataset.yaml
```

It points to the relative `archive` directory and uses the standard YOLO detection format with `train`, `valid`, and `test` splits.

## Run Experiments

Check the experiment matrix without training:

```powershell
python scripts/run_experiments.py --dry-run
```

Run comparison experiments:

```powershell
python scripts/run_experiments.py --group comparison
```

Run ablation experiments:

```powershell
python scripts/run_experiments.py --group ablation
```

Run the single-channel spatial-attention model directly:

```powershell
python train_yolo11.py --device 0 --workers 0 --name single_channel_spatial
```

Collect paper-ready metrics:

```powershell
python scripts/summarize_results.py --runs-dir runs/paper --out paper_results.xlsx
```

## Experiment Design

Comparison experiments compare YOLO model families or sizes under the same data split, image size, epochs, optimizer, seed, and augmentation policy.

Ablation experiments keep the base model fixed and change only one factor at a time. The starter matrix includes:

- baseline
- no pretrained weights
- no mosaic augmentation
- larger input resolution

Replace or extend `configs/experiments.yaml` with the actual improved modules used by the paper.
