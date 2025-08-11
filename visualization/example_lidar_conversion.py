#!/usr/bin/env python3
"""
Example script showing how to use the LiDAR conversion utilities.

This script demonstrates various ways to convert NuScenes LiDAR data to PLY format.
"""

import os
import sys

# Add the parent directory to path so we can import from visualization
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.lidar_utils import scene_to_ply, load_scene_lidar_data, write_ply_file


def example_basic_conversion():
    """Example: Basic scene conversion to PLY"""
    data_path = "/path/to/your/nuscenes/processed/scene/000"  # Update this path
    output_path = "scene_000_basic.ply"
    
    print("=== Basic Scene Conversion ===")
    try:
        scene_to_ply(
            data_path=data_path,
            output_path=output_path,
            color_by_frame=True,
            apply_camera_alignment=True
        )
        print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_frame_range_conversion():
    """Example: Convert only specific frame range"""
    data_path = "/path/to/your/nuscenes/processed/scene/000"  # Update this path
    output_path = "scene_000_frames_10_20.ply"
    
    print("\n=== Frame Range Conversion (frames 10-20) ===")
    try:
        scene_to_ply(
            data_path=data_path,
            output_path=output_path,
            start_frame=10,
            end_frame=20,
            color_by_frame=True,
            apply_camera_alignment=True
        )
        print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_custom_processing():
    """Example: Load data and do custom processing before saving"""
    data_path = "/path/to/your/nuscenes/processed/scene/000"  # Update this path
    output_path = "scene_000_custom.ply"
    
    print("\n=== Custom Processing Example ===")
    try:
        # Load the raw point data
        points, frame_indices = load_scene_lidar_data(
            data_path=data_path,
            start_frame=0,
            end_frame=10,  # Only first 10 frames
            apply_camera_alignment=True
        )
        
        print(f"Loaded {len(points):,} points from {len(set(frame_indices))} frames")
        
        # Example: Filter points by height (remove ground and high points)
        height_mask = (points[:, 2] > -2.0) & (points[:, 2] < 5.0)
        filtered_points = points[height_mask]
        filtered_frames = frame_indices[height_mask]
        
        print(f"After height filtering: {len(filtered_points):,} points")
        
        # Example: Create custom colors based on height
        import numpy as np
        heights = filtered_points[:, 2]
        height_normalized = (heights - heights.min()) / (heights.max() - heights.min())
        
        # Create RGB colors: blue (low) to red (high)
        colors = np.zeros((len(filtered_points), 3), dtype=np.uint8)
        colors[:, 0] = (height_normalized * 255).astype(np.uint8)  # Red channel
        colors[:, 2] = ((1 - height_normalized) * 255).astype(np.uint8)  # Blue channel
        
        # Save with custom colors
        write_ply_file(filtered_points, colors, output_path)
        print(f"✓ Saved custom processed PLY: {output_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


def example_no_alignment():
    """Example: Convert without camera alignment"""
    data_path = "/path/to/your/nuscenes/processed/scene/000"  # Update this path
    output_path = "scene_000_no_alignment.ply"
    
    print("\n=== No Camera Alignment Example ===")
    try:
        scene_to_ply(
            data_path=data_path,
            output_path=output_path,
            color_by_frame=False,  # Single color
            apply_camera_alignment=False  # No alignment
        )
        print(f"✓ Saved: {output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")


def main():
    print("NuScenes LiDAR to PLY Conversion Examples")
    print("=" * 50)
    
    # Check for available data
    example_data_path = "/workspace/drivestudio/data/nuscenes/processed_10Hz/mini/000"

    if example_data_path:
        print(f"\n🎉 Found example data at: {example_data_path}")
        print("Running conversion example...")
        
        # Run a quick example
        try:
            scene_to_ply(
                data_path=example_data_path,
                output_path="lidar_source.ply",
                start_frame=75,
                end_frame=125,  # Just first 5 frames for quick demo
                color_by_frame=True,
                apply_camera_alignment=True
            )
            print("✓ Success! Check 'example_scene.ply'")
        except Exception as e:
            print(f"✗ Example failed: {e}")
    else:
        print("\n⚠️  No example data found.")
        print("To use this script:")
        print("1. Update the data_path variables in the example functions")
        print("2. Make sure you have processed NuScenes data with lidar/ and lidar_pose/ directories")
        print("3. Run the individual example functions")


if __name__ == "__main__":
    main()
