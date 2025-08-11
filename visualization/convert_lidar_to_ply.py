#!/usr/bin/env python3
"""
Convert NuScenes LiDAR .bin files to a single .ply file using world coordinate alignment.

This script reads processed NuScenes LiDAR data (.bin files) and their corresponding
pose files, transforms all points to world coordinates using the same alignment method
as the sourceloader, and saves them as a single .ply file for visualization.

Usage:
    python convert_lidar_to_ply.py --data_path /path/to/scene --output_path output.ply [--start_frame 0] [--end_frame -1]

Example:
    python convert_lidar_to_ply.py --data_path /data/nuscenes/processed/mini/000 --output_path scene_000.ply
"""

import argparse
import os
from typing import Optional

import numpy as np
from tqdm import trange


def write_ply(points: np.ndarray, colors: Optional[np.ndarray], filename: str):
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


def load_lidar_frame(lidar_file: str, pose_file: str) -> np.ndarray:
    """
    Load a single LiDAR frame and transform to world coordinates.
    
    Args:
        lidar_file: path to .bin file containing ego-frame points
        pose_file: path to .txt file containing lidar-to-world transformation
        
    Returns:
        (N, 3) array of world-coordinate points
    """
    # Load ego-frame LiDAR points
    lidar_info = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    lidar_points = lidar_info[:, :3]  # (N, 3)
    
    # Load lidar-to-world transformation matrix
    lidar_to_world = np.loadtxt(pose_file)  # (4, 4)
    
    # Transform points to world coordinates
    # Add homogeneous coordinate
    lidar_points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])  # (N, 4)
    
    # Transform to world coordinates
    world_points = (lidar_to_world @ lidar_points_homo.T).T  # (N, 4)
    
    return world_points[:, :3]  # Return only XYZ


def generate_frame_colors(num_frames: int) -> np.ndarray:
    """
    Generate distinct colors for each frame.
    
    Args:
        num_frames: number of frames
        
    Returns:
        (num_frames, 3) array of RGB colors (0-255)
    """
    import matplotlib.cm as cm
    
    # Use a colormap to generate distinct colors
    cmap = cm.get_cmap('tab10' if num_frames <= 10 else 'tab20')
    colors = []
    
    for i in range(num_frames):
        color = cmap(i / max(1, num_frames - 1))[:3]  # Get RGB, ignore alpha
        colors.append([int(c * 255) for c in color])
    
    return np.array(colors)


def convert_lidar_to_ply(
    data_path: str, 
    output_path: str,
    start_frame: int = 0,
    end_frame: int = -1,
    color_by_frame: bool = True,
    apply_alignment: bool = True
):
    """
    Convert NuScenes LiDAR .bin files to a single .ply file.
    
    Args:
        data_path: path to processed scene directory containing lidar/ and lidar_pose/ subdirs
        output_path: output .ply file path
        start_frame: first frame to include (default: 0)
        end_frame: last frame to include (default: -1 for all frames)
        color_by_frame: whether to color points by frame number
        apply_alignment: whether to apply first-camera alignment (same as sourceloader)
    """
    lidar_dir = os.path.join(data_path, "lidar")
    pose_dir = os.path.join(data_path, "lidar_pose")
    extrinsics_dir = os.path.join(data_path, "extrinsics")
    
    # Check if directories exist
    if not os.path.exists(lidar_dir):
        raise FileNotFoundError(f"LiDAR directory not found: {lidar_dir}")
    if not os.path.exists(pose_dir):
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")
    
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
    
    # Load first camera pose for alignment (same as sourceloader)
    camera_front_start = None
    if apply_alignment and os.path.exists(extrinsics_dir):
        try:
            camera_front_start = np.loadtxt(
                os.path.join(extrinsics_dir, f"{start_frame:03d}_0.txt")
            )
            print("Applying camera alignment using first front camera pose")
        except FileNotFoundError:
            print("Warning: Could not load first camera pose, skipping alignment")
            apply_alignment = False
    else:
        apply_alignment = False
    
    # Generate colors for frames if needed
    frame_colors = None
    if color_by_frame:
        frame_colors = generate_frame_colors(end_frame - start_frame)
    
    # Process all frames
    all_points = []
    all_colors = []
    total_points = 0
    
    for frame_idx in trange(start_frame, end_frame, desc="Processing frames"):
        # File paths
        lidar_file = os.path.join(lidar_dir, f"{frame_idx:03d}.bin")
        pose_file = os.path.join(pose_dir, f"{frame_idx:03d}.txt")
        
        if not os.path.exists(lidar_file):
            print(f"Warning: LiDAR file not found: {lidar_file}")
            continue
        if not os.path.exists(pose_file):
            print(f"Warning: Pose file not found: {pose_file}")
            continue
        
        # Load and transform points
        world_points = load_lidar_frame(lidar_file, pose_file)
        
        # Apply alignment if requested (same as sourceloader)
        if apply_alignment and camera_front_start is not None:
            # Transform points using inverse of first camera pose
            world_points_homo = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
            aligned_points = (np.linalg.inv(camera_front_start) @ world_points_homo.T).T
            world_points = aligned_points[:, :3]
        
        all_points.append(world_points)
        
        # Assign colors if requested
        if color_by_frame:
            frame_color = frame_colors[frame_idx - start_frame]
            point_colors = np.tile(frame_color, (world_points.shape[0], 1))
            all_colors.append(point_colors)
        
        total_points += world_points.shape[0]
    
    if not all_points:
        raise RuntimeError("No valid frames found")
    
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors) if color_by_frame else None
    
    print(f"Total points: {total_points:,}")
    print(f"Point cloud bounds:")
    print(f"  X: [{combined_points[:, 0].min():.2f}, {combined_points[:, 0].max():.2f}]")
    print(f"  Y: [{combined_points[:, 1].min():.2f}, {combined_points[:, 1].max():.2f}]")
    print(f"  Z: [{combined_points[:, 2].min():.2f}, {combined_points[:, 2].max():.2f}]")
    
    # Save as PLY
    print(f"Saving to {output_path}")
    write_ply(combined_points, combined_colors, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert NuScenes LiDAR .bin files to .ply")
    parser.add_argument("--data_path", required=True, 
                       help="Path to processed scene directory (containing lidar/ and lidar_pose/)")
    parser.add_argument("--output_path", required=True,
                       help="Output .ply file path")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="First frame to include (default: 0)")
    parser.add_argument("--end_frame", type=int, default=-1,
                       help="Last frame to include (default: -1 for all)")
    parser.add_argument("--no_color", action="store_true",
                       help="Don't color points by frame")
    parser.add_argument("--no_alignment", action="store_true",
                       help="Don't apply camera alignment (same as sourceloader)")
    
    args = parser.parse_args()
    
    convert_lidar_to_ply(
        data_path=args.data_path,
        output_path=args.output_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        color_by_frame=not args.no_color,
        apply_alignment=not args.no_alignment
    )


if __name__ == "__main__":
    main()
