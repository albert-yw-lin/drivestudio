# Docker Setup

Docker environment for drivestudio (2DGS + 3DGS) development and training.

## Image
- Base: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`
- Python: 3.9
- Torch: 2.0.0 + CUDA 11.8 wheels
- Includes: xformers, pytorch3d, nvdiffrast, Open3D 0.14.1 (TSDF compatibility)

## Custom gsplat Fork
Installed via:
```
RUN python3.9 -m pip install git+https://github.com/albert-yw-lin/gsplat_2dgs.git
```
This uses a temporary fork to patch a bug in the upstream `gsplat`. A PR has been opened; once merged, this line can be switched back to the official repository.

## Verified Runtime Environment
Confirmed working on an HPC node with 4× NVIDIA A100 40GB (driver 570.124.06, CUDA 12.8 runtime on host). Sample `nvidia-smi` excerpt at verification time:
```
GPU  Name                 Temp  Perf  Pwr  Memory-Usage
0    NVIDIA A100-40GB     34C   P0    47W  18701MiB / 40960MiB
1    NVIDIA A100-40GB     34C   P0    45W      4MiB / 40960MiB
2    NVIDIA A100-40GB     32C   P0    46W      4MiB / 40960MiB
3    NVIDIA A100-40GB     32C   P0    46W      4MiB / 40960MiB
Driver: 570.124.06  |  CUDA: 12.8 (host)  |  Container CUDA base: 11.8
```

## Notes
- No conda environment; pure system Python 3.9.
- Open3D pinned at 0.14.1 due to later TSDF API removals.