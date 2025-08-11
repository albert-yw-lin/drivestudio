"""
Utility functions for converting LiDAR data to PLY format.

This module provides functions to convert NuScenes processed LiDAR .bin files
to PLY format using the same world coordinate alignment as the sourceloader.
"""

import os
from typing import Optional, Tuple

import numpy as np
from tqdm import trange


def write_ply_file(points: np.ndarray, colors: Optional[np.ndarray], filename: str) -> None:
    """
    Write point cloud to PLY format.
    
    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255), optional
        filename: output filename
    """
    N = points.shape[0]
    
    # PLY header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y", 
        "property float z",
    ]
    
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green", 
            "property uchar blue"
        ])
    
    header.append("end_header")
    
    # Write file
    with open(filename, 'w') as f:
        # Write header
        for line in header:
            f.write(line + '\n')
        
        # Write points
        for i in range(N):
            if colors is not None:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                       f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
            else:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")


def load_and_transform_lidar_frame(
    lidar_file: str, 
    pose_file: str, 
    camera_alignment: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Load a single LiDAR frame and transform to world coordinates.
    
    Args:
        lidar_file: path to .bin file containing ego-frame points
        pose_file: path to .txt file containing lidar-to-world transformation
        camera_alignment: optional 4x4 matrix for camera alignment (same as sourceloader)
        
    Returns:
        (N, 3) array of world-coordinate points
    """
    # Load ego-frame LiDAR points
    lidar_info = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    lidar_points = lidar_info[:, :3]  # (N, 3)
    
    # Load lidar-to-world transformation matrix
    lidar_to_world = np.loadtxt(pose_file)  # (4, 4)
    
    # Apply camera alignment if provided (same as sourceloader)
    if camera_alignment is not None:
        lidar_to_world = np.linalg.inv(camera_alignment) @ lidar_to_world
    
    # Transform points to world coordinates
    # Add homogeneous coordinate
    lidar_points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])  # (N, 4)
    
    # Transform to world coordinates
    world_points = (lidar_to_world @ lidar_points_homo.T).T  # (N, 4)
    
    return world_points[:, :3]  # Return only XYZ


def load_scene_lidar_data(
    data_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    apply_camera_alignment: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load all LiDAR data for a scene and transform to world coordinates.
    
    Args:
        data_path: path to processed scene directory
        start_frame: first frame to include
        end_frame: last frame to include (None for all frames)
        apply_camera_alignment: whether to apply camera alignment
        
    Returns:
        Tuple of (points, frame_indices) where:
        - points: (N, 3) array of all world-coordinate points
        - frame_indices: (N,) array indicating which frame each point came from
    """
    lidar_dir = os.path.join(data_path, "lidar")
    pose_dir = os.path.join(data_path, "lidar_pose")
    extrinsics_dir = os.path.join(data_path, "extrinsics")
    
    # Check directories exist
    if not os.path.exists(lidar_dir):
        raise FileNotFoundError(f"LiDAR directory not found: {lidar_dir}")
    if not os.path.exists(pose_dir):
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")
    
    # Get available frames
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])
    if not lidar_files:
        raise FileNotFoundError(f"No .bin files found in {lidar_dir}")
    
    total_frames = len(lidar_files)
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    start_frame = max(0, start_frame)
    
    # Load camera alignment if requested
    camera_alignment = None
    if apply_camera_alignment and os.path.exists(extrinsics_dir):
        try:
            camera_alignment = np.loadtxt(
                os.path.join(extrinsics_dir, f"{start_frame:03d}_0.txt")
            )
        except FileNotFoundError:
            print("Warning: Could not load first camera pose for alignment")
    
    # Load all frames
    all_points = []
    all_frame_indices = []
    
    for frame_idx in trange(start_frame, end_frame, desc="Loading LiDAR frames"):
        lidar_file = os.path.join(lidar_dir, f"{frame_idx:03d}.bin")
        pose_file = os.path.join(pose_dir, f"{frame_idx:03d}.txt")
        
        if not (os.path.exists(lidar_file) and os.path.exists(pose_file)):
            continue
        
        # Load and transform points
        world_points = load_and_transform_lidar_frame(
            lidar_file, pose_file, camera_alignment
        )
        
        all_points.append(world_points)
        all_frame_indices.append(np.full(len(world_points), frame_idx))
    
    if not all_points:
        raise RuntimeError("No valid frames found")
    
    # Combine all data
    combined_points = np.vstack(all_points)
    combined_frame_indices = np.hstack(all_frame_indices)
    
    return combined_points, combined_frame_indices


def scene_to_ply(
    data_path: str,
    output_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    color_by_frame: bool = True,
    apply_camera_alignment: bool = True
) -> None:
    """
    Convert a NuScenes scene's LiDAR data to PLY format.
    
    This function uses the same coordinate transformation and alignment
    as the NuScenesLiDARSource class.
    
    Args:
        data_path: path to processed scene directory
        output_path: output .ply file path
        start_frame: first frame to include
        end_frame: last frame to include (None for all)
        color_by_frame: whether to color points by frame number
        apply_camera_alignment: whether to apply camera alignment
    """
    # Load all LiDAR data
    points, frame_indices = load_scene_lidar_data(
        data_path, start_frame, end_frame, apply_camera_alignment
    )
    
    # Generate colors if requested
    colors = None
    if color_by_frame:
        import matplotlib.cm as cm
        
        unique_frames = np.unique(frame_indices)
        num_frames = len(unique_frames)
        
        # Use colormap for frame colors
        cmap = cm.get_cmap('tab10' if num_frames <= 10 else 'tab20')
        
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        for i, frame_idx in enumerate(unique_frames):
            mask = frame_indices == frame_idx
            color = cmap(i / max(1, num_frames - 1))[:3]
            colors[mask] = [int(c * 255) for c in color]
    
    # Write PLY file
    write_ply_file(points, colors, output_path)
    
    print(f"Saved {len(points):,} points from {len(np.unique(frame_indices))} frames to {output_path}")
    print(f"Point cloud bounds:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
