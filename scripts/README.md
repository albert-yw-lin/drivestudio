# Scripts Overview

This directory contains helper shell scripts for common workflows: environment setup (one legacy helper for conda), preprocessing datasets, training, and evaluation. If possible, prefer the Docker workflow (see `Docker/` and top-level `Dockerfile.drivestudio`) because it provides a stable, tested environment without manually juggling CUDA/toolchain versions.

## IMPORTANT (Legacy Only): `run_before_everything_if_using_conda.sh`
This is the ONLY legacy/experimental script in this folder. All other scripts are current and considered normal entrypoints. This helper was created early for ad‑hoc local conda/mamba setups. Prefer Docker instead. It tries to:
- Export `PYTHONPATH` to the repo root
- Discover a CUDA toolkit either inside the active conda environment (`$CONDA_PREFIX/bin/nvcc`) or fall back to the system `nvcc`
- Set `CUDA_PATH`, `CUDA_HOME`, and minimal `LD_LIBRARY_PATH`, plus a few compiler-related paths

Limitations / Warnings:
- Fragile: small mismatches between PyTorch, CUDA driver, and toolkit versions can cause build or runtime errors.
- Not continuously maintained; may break with driver or toolkit updates.
- Use only if you truly need an ad‑hoc local (non-Docker) run. All other scripts do NOT require this.

Recommended Instead:
- Build and run inside the provided Docker image (`Dockerfile.drivestudio`) for reproducible builds.
- Skip this script entirely when using Docker; the container sets required environment variables.

Quick Use (legacy, NOT recommended long term):
```bash
source scripts/run_before_everything_if_using_conda.sh
```

## `preprocess.sh`
Wrapper to launch dataset preprocessing (calls into `datasets/preprocess.py` or related tooling via configs).

Tunable Config Options (examples; see `preprocess.py` and the main README.md):
- `scene_ids`

## Training Scripts
Purpose-built shortcuts that pass a particular config file (and sometimes demo-friendly settings) to the training entrypoint (typically `tools/train.py`).

- `train_waymo.sh`
- `train_pandaset.sh`
- `train_nuscenes.sh`
- `train_for_demo_waymo.sh` (demo-oriented)
- `train_for_demo_nuscenes_pandaset.sh` (demo-oriented)

Typical pattern if you want to adapt manually:
```bash
python tools/train.py \
  --config configs/omnire.yaml \
  --output outputs/my_experiment \
  --device cuda:0
```

## Evaluation Scripts
- `eval.sh`: Standard evaluation using a chosen config + checkpoint.
- `eval_for_demo.sh`: Streamlined or demo-focused evaluation (likely selects a curated subset of scenes / frames for quicker turnaround).

Example:
```bash
bash scripts/eval.sh \
  --config configs/omnire.yaml \
  --ckpt outputs/my_experiment/checkpoints/epoch_20.pt
```

## Suggested Workflow (Stable)
1. Build Docker image:
   ```bash
   docker build -f Docker/Dockerfile.drivestudio -t drivestudio:latest .
   ```
2. (Optionally) use `docker-compose -f Docker/docker-compose.drivestudio.yml up` if provided.
3. Preprocess datasets inside the container.
4. Train with one of the `train_*.sh` or direct `python tools/train.py` commands.
5. Evaluate with `eval.sh`.

## When to Avoid the Conda Script
Use Docker if:
- You want reproducibility across machines.
- You are not debugging CUDA extensions locally.
- You do not need custom system-level profiling tools outside the container.

Only consider the conda route if:
- You are iterating on low-level CUDA ops and prefer a host install.
- You have matching versions (PyTorch, CUDA toolkit, NVIDIA driver) already aligned.

## Common Pitfalls (Non-Docker)
- LD library conflicts: Having multiple CUDA toolkits / stray `LD_LIBRARY_PATH` entries from other environments.
- Compiler mismatch: PyTorch extensions may expect gcc/g++ 9–11; adjust via `CC` and `CXX` if needed.
- Mixed Python versions: Ensure the environment interpreter matches compiled extension artifacts.

## FAQ
- Q: Do I need to run the conda script before every training run?  
  A: No—only if you are in a bare conda environment without Docker and have not already exported those variables in the current shell.
- Q: Where do I set `scene_ids`?  
  A: In the chosen config YAML or via an override flag if the script supports it.
- Q: Can I mix datasets?  
  A: Use a config that supports multi-dataset training (see combined demo scripts) or create a new YAML referencing multiple dataset blocks.

## Minimal Quick Start (Docker Path)
```bash
# Build image
docker build -f Docker/Dockerfile.drivestudio -t drivestudio:latest .

# Run container (mount datasets + outputs as needed)
docker run --gpus all -it \
  -v /path/to/datasets:/datasets \
  -v $(pwd)/outputs:/workspace/outputs \
  drivestudio:latest /bin/bash

# Inside container: preprocess, then train
bash scripts/preprocess.sh --config configs/omnire.yaml
bash scripts/train_waymo.sh
```

---
For deeper configuration details, inspect specific YAML files under `configs/` and the Python entrypoints in `tools/`.
