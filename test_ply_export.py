#!/usr/bin/env python3
"""
Test script to verify PLY export functionality works with VanillaGaussians model
"""

import torch
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ply_export():
    """Test the export_gaussians_to_ply function with a VanillaGaussians model"""
    
    # Import required modules
    from utils.misc import export_gaussians_to_ply
    from models.gaussians.vanilla import VanillaGaussians
    from omegaconf import OmegaConf
    
    print("Testing PLY export functionality...")
    
    # Create a simple test configuration
    ctrl_cfg = OmegaConf.create({
        'sh_degree': 3,  # Test with higher SH degree to test SH features
        'ball_gaussians': False,
        'gaussian_2d': False,
        'refine_interval': 100,
        'warmup_steps': 500,
        'stop_split_at': 15000,
        'reset_alpha_interval': 3000,
        'densify_grad_thresh': 0.0002,
        'densify_size_thresh': 0.01,
        'n_split_samples': 2,
        'cull_alpha_thresh': 0.1,
        'cull_scale_thresh': 0.5,
        'reset_alpha_value': 0.01,
        'stop_screen_size_at': 4000,
        'sh_degree_interval': 1000
    })
    
    # Create a VanillaGaussians model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaGaussians(
        class_name="Test",
        ctrl=ctrl_cfg,
        scene_scale=10.0,
        scene_origin=torch.zeros(3),
        num_train_images=100,
        device=device
    )
    
    # Initialize with some dummy data
    num_points = 1000
    init_means = torch.randn(num_points, 3, device=device)
    init_colors = torch.rand(num_points, 3, device=device)
    
    model.create_from_pcd(init_means, init_colors)
    
    # Test export
    output_path = "outputs/test"
    os.makedirs(output_path, exist_ok=True)
    
    try:
        export_gaussians_to_ply(model, output_path, "test_gaussians.ply")
        print("✅ PLY export test passed!")
        print(f"Exported PLY file to: {output_path}/test_gaussians.ply")
        
        # Check if file exists and has reasonable size
        ply_file = os.path.join(output_path, "test_gaussians.ply")
        if os.path.exists(ply_file):
            file_size = os.path.getsize(ply_file)
            print(f"PLY file size: {file_size} bytes")
            if file_size > 0:
                print("✅ PLY file created successfully with non-zero size!")
                return True
            else:
                print("❌ PLY file is empty!")
                return False
        else:
            print("❌ PLY file was not created!")
            return False
            
    except Exception as e:
        print(f"❌ PLY export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ply_export()
