# NuScenes LiDAR to PLY Conversion

This directory contains utilities to convert processed NuScenes LiDAR data (.bin files) to PLY format for visualization. The conversion uses the same world coordinate alignment method as the `NuScenesLiDARSource` class in the sourceloader.

## Files

- `convert_lidar_to_ply.py` - Command-line script for converting scenes
- `lidar_utils.py` - Utility functions that can be imported
- `example_lidar_conversion.py` - Example usage and demonstrations
- `README.md` - This documentation

## Quick Start

### Command Line Usage

```bash
# Convert entire scene
python convert_lidar_to_ply.py --data_path /path/to/scene/000 --output_path scene_000.ply

# Convert specific frame range
python convert_lidar_to_ply.py --data_path /path/to/scene/000 --output_path scene_000_frames_10_20.ply --start_frame 10 --end_frame 20

# Convert without frame colors
python convert_lidar_to_ply.py --data_path /path/to/scene/000 --output_path scene_000_no_color.ply --no_color

# Convert without camera alignment
python convert_lidar_to_ply.py --data_path /path/to/scene/000 --output_path scene_000_no_align.ply --no_alignment
```

### Python API Usage

```python
from visualization.lidar_utils import scene_to_ply, load_scene_lidar_data

# Simple conversion
scene_to_ply(
    data_path="/path/to/scene/000",
    output_path="output.ply",
    color_by_frame=True,
    apply_camera_alignment=True
)

# Load data for custom processing
points, frame_indices = load_scene_lidar_data(
    data_path="/path/to/scene/000",
    start_frame=0,
    end_frame=10
)
```

## Data Requirements

The input `data_path` should be a processed NuScenes scene directory containing:

```
scene_directory/
├── lidar/           # .bin files with ego-frame point clouds
│   ├── 000.bin
│   ├── 001.bin
│   └── ...
├── lidar_pose/      # .txt files with lidar-to-world transformation matrices
│   ├── 000.txt
│   ├── 001.txt
│   └── ...
└── extrinsics/      # (optional) camera extrinsics for alignment
    ├── 000_0.txt    # front camera pose for frame 000
    └── ...
```

## Coordinate System

### Input Data
- **LiDAR .bin files**: Points in ego vehicle coordinate system (4 columns: x, y, z, intensity)
- **Pose .txt files**: 4x4 transformation matrices from lidar sensor to world coordinates

### Output PLY
- **Without alignment**: Points in NuScenes world coordinate system
- **With alignment** (default): Points aligned to first front camera pose (same as sourceloader)

The alignment transformation matches exactly what `NuScenesLiDARSource` does:

```python
# Load first camera pose
camera_front_start = np.loadtxt("extrinsics/000_0.txt")

# Apply to each lidar pose
lidar_to_world_aligned = np.linalg.inv(camera_front_start) @ lidar_to_world_original
```

## Visualization

The generated PLY files can be viewed in:
- **CloudCompare** (recommended for large point clouds)
- **MeshLab**
- **Open3D viewer**
- **Blender**

### Color Coding

When `color_by_frame=True` (default):
- Each frame gets a distinct color from a colormap
- Useful for visualizing temporal progression
- Colors cycle through tab10/tab20 colormaps

## Examples

### Basic Scene Conversion
```python
from visualization.lidar_utils import scene_to_ply

scene_to_ply(
    data_path="data/nuscenes/processed/mini/000",
    output_path="scene_000.ply"
)
```

### Custom Processing
```python
from visualization.lidar_utils import load_scene_lidar_data, write_ply_file
import numpy as np

# Load data
points, frame_indices = load_scene_lidar_data("data/nuscenes/processed/mini/000")

# Filter by height
mask = (points[:, 2] > -1.5) & (points[:, 2] < 3.0)
filtered_points = points[mask]

# Create height-based colors
heights = filtered_points[:, 2]
height_norm = (heights - heights.min()) / (heights.max() - heights.min())
colors = np.zeros((len(filtered_points), 3), dtype=np.uint8)
colors[:, 0] = (height_norm * 255).astype(np.uint8)  # Red for high
colors[:, 2] = ((1 - height_norm) * 255).astype(np.uint8)  # Blue for low

# Save
write_ply_file(filtered_points, colors, "scene_height_colored.ply")
```

## Performance Notes

- Processing time scales with number of frames and points per frame
- Typical NuScenes scene (20 frames): ~1-2 minutes, ~1M points, ~50MB PLY file
- Memory usage: ~8 bytes per point (for coordinates only)
- Large scenes may require chunked processing for memory efficiency

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that `data_path` contains `lidar/` and `lidar_pose/` directories
2. **No valid frames found**: Ensure .bin and .txt files have matching frame numbers
3. **Memory error**: Reduce frame range or process in chunks for very large scenes

### Debugging

Run the example script to check for available data:
```bash
python example_lidar_conversion.py
```

This will automatically search for processed NuScenes data and run a test conversion.

## Related Files

- `datasets/nuscenes/nuscenes_sourceloader.py` - Reference implementation for coordinate transformations
- `datasets/nuscenes/nuscenes_preprocess.py` - Creates the input .bin and pose files
