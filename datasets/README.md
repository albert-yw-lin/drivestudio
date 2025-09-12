# Datasets Module

This directory contains all dataset-related logic: preprocessing scripts, dataset metadata, source loaders, and utility tools for handling multi-sensor autonomous driving datasets.

## Purpose
- Preprocessing raw datasets into the unified processed format used by training.
- Defining dataset splits, metadata, and per-frame access (`driving_dataset.py`).
- Implementing per-dataset source loaders (images, LiDAR, poses, annotations).
- Human pose extraction / SMPL processing utilities.
- Auxiliary conversion / visualization helpers under `tools/` and dataset subfolders.

Supported datasets: Waymo, NuScenes, ArgoVerse, PandaSet, KITTI, NuPlan (see matching subfolders).

## Key Files
- `dataset_meta.py`: Central metadata & dataset registry.
- `driving_dataset.py`: High-level dataset wrapper used by trainers.
- `preprocess.py`: Entry for multi-dataset preprocessing orchestration.
- `base/`: Generic abstractions (`scene_dataset`, `lidar_source`, `pixel_source`, split utilities).
- `<dataset_name>_preprocess.py`: Per-dataset preprocessing pipeline.
- `<dataset_name>_sourceloader.py`: Runtime frame loading (images, intrinsics, poses, LiDAR).
- `tools/`: Postprocessing helpers (mask extraction, SMPL extraction, human pose merging, etc.).

## Notes
- NuScenes: `apply_calibration_offset` helper (in `nuscenes_sourceloader.py`) is present but commented out; uncomment to experiment with pose perturbation + `CameraOptModule`.
- Waymo: `waymo_combine_lidar_files.py` merges per-frame LiDAR into a single PLY using ego-pose alignment.
 - Waymo intensity: current combined output shows intensity values 0 or 1 in produced PLY (observation only). This might be an implementation bug or just lack of inensity in the data. Both Nuscenes and Waymo have the same issue. Did not try other dataset.
 - Visualization: root `visualization/convert_lidar_to_ply.py` provides similar functionality (verified on NuScenes); dataset formats differ, so other datasets may not load directly.