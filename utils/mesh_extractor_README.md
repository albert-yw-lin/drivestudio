# Mesh Extractor Arguments Guide

This document summarizes the key tunable arguments and parameters in `utils/mesh_extractor.py`, which provides background depth extraction and TSDF mesh fusion (Open3D CPU / GPU) for the Background Gaussian model.

## Classes
- **`SimpleCam`**: Lightweight container for camera-to-world transform `camtoworlds (4x4)`, intrinsics matrix `Ks (3x3)`, and image height `H`, width `W`.
- **`BackgroundMeshExtractor`**: Orchestrates background depth rendering from a trainer and TSDF fusion into a mesh.

## Initialization
| Argument | Default | Purpose | Tips |
|----------|---------|---------|------|
| `scene_scale` | `30.0` | Approximate global scale (meters) used implicitly when reasoning about scene extent. | If scenes are very small (<10m) or huge (>100m), adjust to keep TSDF truncation and voxel sizing meaningful.

## `extract_background_depths_from_trainer()`
Extracts per‑view background depth & RGB from a multi-class trainer.

| Argument | Default | Purpose | Guidance |
|----------|---------|---------|----------|
| `trainer` | (req.) | MultiTrainer-like object; must produce `Background_depth` (and optionally `Background_rgb`, `rgb_sky`, `Background_opacity`). | Ensure `trainer.render_each_class` is supported. | 
| `dataset` | (req.) | Dataset with `full_image_set.get_image(idx, downscale)` returning `(image_infos, cam_infos)`. | Must supply intrinsics & pose tensors. |
| `num_views` | `None` | Limit number of evenly sampled views; `None` = all. | 60–150 good for mid-quality; >200 for cleaner fusion; fewer speeds things up.
| `camera_downscale` | `2` | Downscale factor for rendering. | Larger values (e.g., 4) reduce memory/time but coarser mesh; 1 for highest fidelity.

### Internal Behavior
- Forces `trainer.render_each_class = True` temporarily.
- Expects `Background_depth` in outputs; raises if missing.
- Blends sky RGB if `rgb_sky` and `Background_opacity` are provided.
- Stores camera extrinsics (`camera_to_world`) and intrinsics for later TSDF integration.
- Derives a bounding sphere (`center`, `radius`) from camera centers.

## `extract_mesh_tsdf()`
Fuses stored depths into a TSDF and extracts a triangle mesh. Supports Open3D Tensor (GPU) if available; otherwise falls back to legacy CPU `ScalableTSDFVolume`.

| Argument | Default | Purpose | Tuning Advice |
|----------|---------|---------|---------------|
| `voxel_size` | `0.05` | Linear size (meters) of voxels. | Most impactful knob: smaller = higher detail & memory cost. Try 0.1 (fast), 0.05 (balanced), 0.02 (high‑detail, heavy). |
| `sdf_trunc` | `voxel_size * 5` | Truncation distance for signed distance values. | Keep between 3–8× `voxel_size`. Larger smooths surfaces; smaller preserves sharper detail but risks holes. |
| `depth_trunc` | `2 * radius` | Max depth integrated (far clipping). | If large far geometry/noise, lower this (e.g., `radius * 1.2`). |
| `depth_threshold` | `None` | Optional additional max depth: values above set to 0 before fusion (remove sky). | Set slightly below sky / infinity depths (e.g., 80–120m in driving scenes). |
| `use_cuda` | `True` | Enable GPU TSDF if Open3D Tensor and CUDA available. | Disable if GPU memory constrained or debugging differences. |
| `block_resolution` | `16` | Voxel block edge length (GPU backend). | Usually fine; increase (e.g., 32) for slightly fewer kernel launches; keep power of two. |
| `block_count` | `400000` | Allocation limit for voxel blocks (GPU). | Increase for large scenes / small voxels (watch GPU memory). |

### Performance / Memory Notes
- GPU TSDF stores voxel blocks sparsely; memory ~ `block_count * block_resolution^3` occupancy dependent.
- CPU `ScalableTSDFVolume` may be slower but simpler to debug; memory roughly scales with surface area & integration path.

## `post_process_mesh()`
Removes small disconnected triangle clusters and degenerate geometry.

| Argument | Default | Purpose | Tuning |
|----------|---------|---------|--------|
| `min_cluster_size` | `30` | Minimum triangle count for a connected component to be kept. | Increase (e.g. 100–300) to prune floaters; decrease (10) to keep small objects. |

## Typical Workflow
```python
from utils.mesh_extractor import BackgroundMeshExtractor

# 1. Initialize
extractor = BackgroundMeshExtractor(scene_scale=30.0)

# 2. Gather background depths (choose subset of views)
extractor.extract_background_depths_from_trainer(
    trainer=trainer,
    dataset=dataset,
    num_views=120,
    camera_downscale=2,
)

# 3. Fuse to mesh (GPU if possible)
mesh = extractor.extract_mesh_tsdf(
    voxel_size=0.05,
    sdf_trunc=None,        # auto -> voxel_size*5
    depth_trunc=None,      # auto -> 2*radius
    depth_threshold=90.0,  # filter sky (optional)
    use_cuda=True,
)

# 4. Post-process
mesh_clean = extractor.post_process_mesh(mesh, min_cluster_size=50)

# 5. Save mesh
import open3d as o3d
o3d.io.write_triangle_mesh('background_mesh.ply', mesh_clean)
```

## Quick Tuning Cheat Sheet
| Goal | Adjust |
|------|--------|
| Faster draft mesh | Increase `voxel_size` (0.08–0.12), reduce `num_views`, raise `camera_downscale` (3–4). |
| Higher detail | Decrease `voxel_size` (0.02–0.04), keep `sdf_trunc` ~5× voxel, ensure enough views (>=150). |
| Remove sky artifacts | Set `depth_threshold` slightly below max valid depth; reduce `depth_trunc`. |
| Remove floaters | Increase `min_cluster_size` and possibly increase `sdf_trunc` a bit. |
| Preserve thin structures | Lower `voxel_size`; do not set `sdf_trunc` too small (<3× voxel). |
| Reduce memory (GPU) | Increase `voxel_size`; decrease `block_count`; limit `num_views`. |

## Debug Tips
- Print `self.radius` after extraction to validate scene scale; unexpected large radius suggests camera pose error.
- Visualize a few depth maps to ensure sky is far (large depth values) before applying thresholds.
- If mesh is empty on GPU path, verify `depth_threshold` not too small and intrinsics/extrinsics are correct.
- Compare GPU vs CPU outputs by disabling `use_cuda` to isolate backend issues.