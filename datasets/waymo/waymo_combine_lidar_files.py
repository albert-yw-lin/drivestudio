#!/usr/bin/env python3
"""
Convert Waymo LiDAR .bin files to a single .ply file using world coordinate alignment.

This script reads processed Waymo LiDAR data (.bin files) and their corresponding
ego pose files, transforms all points to world coordinates using the same alignment method
as the sourceloader, and saves them as a single .ply file for visualization.

The key differences from NuScenes:
- Uses 'ego_pose' directory instead of 'lidar_pose'
- Extrinsics are ego-relative (only 5 files: 0-4 for cameras) instead of world-relative
- LiDAR data format: [origins(3), points(3), flows(3), ground_label(1), intensity(1), elongation(1), laser_ids(1)] = 14 values per point
- Uses first ego pose for alignment instead of first camera pose

Usage:
    python waymo_combine_lidar_files.py --data_path /path/to/scene [--output_path output.ply] [--start_frame 0] [--end_frame -1]

Example:
    python waymo_combine_lidar_files.py --data_path /data/waymo/processed/000
"""

import argparse
import os
from typing import Optional

import numpy as np
from tqdm import trange


def write_ply(points: np.ndarray, colors: Optional[np.ndarray], filename: str, intensities: Optional[np.ndarray] = None):
    """
    Write point cloud to PLY format.
    
    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (0-255), optional
        intensities: (N,) array of per-point intensities, optional
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
    
    # Choose attribute mode: RGB or single intensity (intensity takes precedence if provided)
    use_intensity = intensities is not None
    use_colors = (colors is not None) and not use_intensity
    
    if use_colors:
        header.extend([
            "property uchar red",
            "property uchar green", 
            "property uchar blue"
        ])
    elif use_intensity:
        header.extend([
            "property float intensity"
        ])
    
    header.append("end_header")
    
    # Write file
    with open(filename, 'w') as f:
        # Write header
        for line in header:
            f.write(line + '\n')
        
        # Write points
        for i in range(N):
            if use_colors:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                       f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
            elif use_intensity:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {float(intensities[i]):.6f}\n")
            else:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")


def load_waymo_lidar_frame(lidar_file: str, ego_pose_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single Waymo LiDAR frame and transform to world coordinates.
    
    Waymo LiDAR .bin format (14 values per point):
    - origins (3): LiDAR ray origins in ego frame
    - points (3): LiDAR points in ego frame  
    - flows (3): scene flow vectors
    - ground_label (1): ground segmentation label
    - intensity (1): point intensity
    - elongation (1): point elongation
    - laser_ids (1): which laser sensor
    
    Args:
        lidar_file: path to .bin file containing ego-frame points (14 values per point)
        ego_pose_file: path to .txt file containing ego-to-world transformation
        
    Returns:
        Tuple of:
        - (N, 3) array of world-coordinate points
        - (N,) array of per-point intensities
    """
    # Load Waymo LiDAR data (14 values per point)
    lidar_data = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 14)
    
    # Extract components according to Waymo format:
    # [origins(3), points(3), flows(3), ground_label(1), intensity(1), elongation(1), laser_ids(1)]
    origins = lidar_data[:, 0:3]      # Ray origins in ego frame
    points = lidar_data[:, 3:6]       # Points in ego frame
    flows = lidar_data[:, 6:9]        # Scene flow vectors
    ground_label = lidar_data[:, 9]   # Ground labels
    intensity = lidar_data[:, 10]     # Intensity values
    elongation = lidar_data[:, 11]    # Elongation values
    laser_ids = lidar_data[:, 12]     # Laser sensor IDs
    
    # Load ego-to-world transformation matrix
    ego_to_world = np.loadtxt(ego_pose_file)  # (4, 4)
    
    # Transform points to world coordinates
    # Add homogeneous coordinate
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
    
    # Transform to world coordinates
    world_points = (ego_to_world @ points_homo.T).T  # (N, 4)
    
    return world_points[:, :3], intensity  # Return XYZ and intensity


def convert_waymo_lidar_to_ply(
    data_path: str, 
    output_path: str,
    start_frame: int = 0,
    end_frame: int = -1,
    enable_color: bool = True,
    intensity_only: bool = False,
    apply_alignment: bool = True
):
    """
    Convert Waymo LiDAR .bin files to a single .ply file.
    
    Args:
        data_path: path to processed scene directory containing lidar/ and ego_pose/ subdirs
        output_path: output .ply file path
        start_frame: first frame to include (default: 0)
        end_frame: last frame to include (default: -1 for all frames)
        enable_color: whether to write RGB colors derived from intensity (ignored if intensity_only)
        intensity_only: whether to write a single float intensity per point instead of RGB
        apply_alignment: whether to apply first-ego-pose alignment (same as sourceloader)
    """
    lidar_dir = os.path.join(data_path, "lidar")
    ego_pose_dir = os.path.join(data_path, "ego_pose")
    
    # Check if directories exist
    if not os.path.exists(lidar_dir):
        raise FileNotFoundError(f"LiDAR directory not found: {lidar_dir}")
    if not os.path.exists(ego_pose_dir):
        raise FileNotFoundError(f"Ego pose directory not found: {ego_pose_dir}")
    
    # Get list of available frames
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])
    if not lidar_files:
        raise FileNotFoundError(f"No .bin files found in {lidar_dir}")
    
    # Determine frame range
    total_frames = len(lidar_files)
    if end_frame == -1:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    start_frame = max(0, start_frame)
    
    if start_frame >= end_frame:
        raise ValueError(f"Invalid frame range: start={start_frame}, end={end_frame}")
    
    print(f"Processing frames {start_frame} to {end_frame-1} ({end_frame - start_frame} frames)")
    
    # Load first ego pose for alignment (same as sourceloader approach)
    ego_start_pose = None
    if apply_alignment:
        try:
            ego_start_pose = np.loadtxt(
                os.path.join(ego_pose_dir, f"{start_frame:03d}.txt")
            )
            print("Applying ego pose alignment using first ego pose")
        except FileNotFoundError:
            print("Warning: Could not load first ego pose, skipping alignment")
            apply_alignment = False
    
    # Process all frames
    all_points = []
    all_intensities = []
    total_points = 0
    
    for frame_idx in trange(start_frame, end_frame, desc="Processing frames"):
        # File paths
        lidar_file = os.path.join(lidar_dir, f"{frame_idx:03d}.bin")
        ego_pose_file = os.path.join(ego_pose_dir, f"{frame_idx:03d}.txt")
        
        if not os.path.exists(lidar_file):
            print(f"Warning: LiDAR file not found: {lidar_file}")
            continue
        if not os.path.exists(ego_pose_file):
            print(f"Warning: Ego pose file not found: {ego_pose_file}")
            continue
        
        # Load and transform points (and intensities)
        world_points, intensities = load_waymo_lidar_frame(lidar_file, ego_pose_file)
        
        # Apply alignment if requested (same as sourceloader: relative to first ego pose)
        if apply_alignment and ego_start_pose is not None:
            # Transform points using inverse of first ego pose (to make first frame origin)
            world_points_homo = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
            aligned_points = (np.linalg.inv(ego_start_pose) @ world_points_homo.T).T
            world_points = aligned_points[:, :3]
        
        all_points.append(world_points)
        all_intensities.append(intensities)
        
        total_points += world_points.shape[0]
    
    if not all_points:
        raise RuntimeError("No valid frames found")
    
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_intensities = np.concatenate(all_intensities)

    # Compute colors from intensity if enabled (unless intensity_only)
    if enable_color and not intensity_only:
        i_min = float(combined_intensities.min()) if combined_intensities.size > 0 else 0.0
        i_max = float(combined_intensities.max()) if combined_intensities.size > 0 else 1.0

        if i_max <= 1.5 and i_min >= 0.0:
            # Typical normalized [0, 1]
            gray = np.clip(combined_intensities, 0.0, 1.0) * 255.0
        elif i_max <= 255.0 and i_min >= 0.0:
            # Already in [0, 255]
            gray = np.clip(combined_intensities, 0.0, 255.0)
        else:
            # Min-max normalize to [0, 255]
            denom = max(1e-6, i_max - i_min)
            gray = np.clip((combined_intensities - i_min) / denom, 0.0, 1.0) * 255.0

        gray = gray.astype(np.uint8)
        combined_colors = np.stack([gray, gray, gray], axis=1)
    else:
        combined_colors = None
    
    print(f"Total points: {total_points:,}")
    print(f"Intensity range: [{combined_intensities.min():.3f}, {combined_intensities.max():.3f}]")
    print(f"Point cloud bounds:")
    print(f"  X: [{combined_points[:, 0].min():.2f}, {combined_points[:, 0].max():.2f}]")
    print(f"  Y: [{combined_points[:, 1].min():.2f}, {combined_points[:, 1].max():.2f}]")
    print(f"  Z: [{combined_points[:, 2].min():.2f}, {combined_points[:, 2].max():.2f}]")
    
    # Save as PLY
    print(f"Saving to {output_path}")
    write_ply(combined_points, combined_colors, output_path, intensities=combined_intensities if intensity_only else None)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert Waymo LiDAR .bin files to .ply (supports intensity-only or intensity-colored)")
    parser.add_argument("--data_path", required=True, 
                       help="Path to processed scene directory (containing lidar/ and ego_pose/)")
    parser.add_argument("--output_path", required=False, default=None,
                       help="Output .ply file path (default: <data_path>/lidar.ply)")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="First frame to include (default: 0)")
    parser.add_argument("--end_frame", type=int, default=-1,
                       help="Last frame to include (default: -1 for all)")
    parser.add_argument("--no_color", action="store_true",
                       help="Don't write colors (ignore intensity)")
    parser.add_argument("--intensity_only", action="store_true",
                       help="Write a single float intensity per point instead of RGB")
    parser.add_argument("--no_alignment", action="store_true",
                       help="Don't apply ego pose alignment (sourceloader does)")
    
    args = parser.parse_args()
    
    output_path = args.output_path or os.path.join(args.data_path, "lidar.ply")

    convert_waymo_lidar_to_ply(
        data_path=args.data_path,
        output_path=output_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        enable_color=not args.no_color,
        intensity_only=args.intensity_only,
        apply_alignment=not args.no_alignment
    )


if __name__ == "__main__":
    main()