#!/usr/bin/env python3
"""
Script to convert Gaussian Splatting models from DriveStudio PTH format to standard PLY format.

PTH format (DriveStudio): Contains PyTorch parameters like:
- _means: [N, 3] - Gaussian centers
- _scales: [N, 3] - Gaussian scales (log space)
- _quats: [N, 4] - Quaternion rotations
- _features_dc: [N, 3] - DC component of spherical harmonics
- _features_rest: [N, (sh_degree+1)^2-1, 3] - Rest of spherical harmonics
- _opacities: [N, 1] - Gaussian opacities (logit space)
- Additional temporal parameters for PVG models: _taus, _betas, _velocity

PLY format (Standard Gaussian Splatting): Contains:
- x, y, z: Positions
- nx, ny, nz: Normals (set to 0)
- f_dc_0, f_dc_1, f_dc_2: DC spherical harmonics
- f_rest_0 to f_rest_44: Rest spherical harmonics
- opacity: Gaussian opacity
- scale_0, scale_1, scale_2: Gaussian scales
- rot_0, rot_1, rot_2, rot_3: Quaternion rotations
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import open3d as o3d
from omegaconf import OmegaConf

# Add the parent directory to the path to import from the repo
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)

def quaternion_normalize(quats: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions to unit length."""
    return quats / quats.norm(dim=-1, keepdim=True)

def load_gaussian_model(pth_path: str, model_name: str = None) -> Dict[str, torch.Tensor]:
    """Load Gaussian model from PTH file."""
    try:
        # Load the checkpoint
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        # Extract model state dict
        if 'models' in checkpoint and isinstance(checkpoint['models'], dict):
            models = checkpoint['models']
            
            # If no specific model name given, show available models and pick the first gaussian model
            available_models = []
            for name, model_dict in models.items():
                if isinstance(model_dict, dict) and '_means' in model_dict:
                    available_models.append(name)
            
            if not available_models:
                raise ValueError("No Gaussian models found in the checkpoint")
            
            if model_name is None:
                model_name = available_models[0]
                logger.info(f"Available models: {available_models}")
                logger.info(f"Using model: {model_name}")
            elif model_name not in available_models:
                raise ValueError(f"Model '{model_name}' not found. Available: {available_models}")
            
            gaussian_params = models[model_name]
            
        else:
            # Try to find gaussian parameters directly in the checkpoint
            gaussian_params = {}
            
            # Look for parameters that match gaussian parameter patterns
            def find_gaussian_params(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if any(param in key for param in ['_means', '_scales', '_quats', '_features_dc', '_features_rest', '_opacities']):
                            param_name = key.split('.')[-1]
                            gaussian_params[param_name] = value
                        elif isinstance(value, dict):
                            find_gaussian_params(value, f"{prefix}.{key}" if prefix else key)
            
            find_gaussian_params(checkpoint)
                
        if not gaussian_params:
            raise ValueError("No Gaussian parameters found in the checkpoint")
            
        return gaussian_params
        
    except Exception as e:
        logger.error(f"Error loading PTH file: {e}")
        raise

def convert_to_ply_format(gaussian_params: Dict[str, torch.Tensor], 
                         opacity_threshold: float = 0.005,
                         normalize_positions: bool = False) -> Dict[str, np.ndarray]:
    """Convert Gaussian parameters to PLY format."""
    
    # Extract basic parameters
    means = gaussian_params['_means']  # [N, 3]
    scales = gaussian_params['_scales']  # [N, 3] or [N, 1]
    quats = gaussian_params['_quats']  # [N, 4]
    features_dc = gaussian_params['_features_dc']  # [N, 3]
    features_rest = gaussian_params['_features_rest']  # [N, sh_bases-1, 3]
    opacities = gaussian_params['_opacities']  # [N, 1]
    
    # Convert to CPU and numpy
    means = means.cpu().numpy()
    scales = scales.cpu().numpy()
    quats = quats.cpu().numpy()
    features_dc = features_dc.cpu().numpy()
    opacities_tensor = torch.tensor(opacities) if not isinstance(opacities, torch.Tensor) else opacities
    opacities = torch.sigmoid(opacities_tensor).cpu().numpy()  # Convert from logit to probability
    
    # Filter by opacity threshold
    valid_mask = opacities.squeeze() > opacity_threshold
    logger.info(f"Filtering {valid_mask.sum()}/{len(valid_mask)} gaussians above opacity threshold {opacity_threshold}")
    
    means = means[valid_mask]
    scales = scales[valid_mask]
    quats = quats[valid_mask]
    features_dc = features_dc[valid_mask]
    opacities = opacities[valid_mask]

    # Normalize quaternions
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    
    # Handle scales - convert from log space and ensure 3D
    scales = np.exp(scales)
    if scales.shape[1] == 1:
        # Ball gaussians - repeat scale for all 3 dimensions
        scales = np.repeat(scales, 3, axis=1)
    elif scales.shape[1] == 2:
        # 2D gaussians - add zero for z dimension
        scales = np.concatenate([scales, np.zeros((scales.shape[0], 1))], axis=1)
    
    # Normalize positions if requested
    assert not normalize_positions
    if normalize_positions:
        center = means.mean(axis=0)
        scale_factor = np.abs(means - center).max() * 1.1
        means = (means - center) / scale_factor
        logger.info(f"Normalized positions: center={center}, scale_factor={scale_factor}")
    
    # Prepare PLY data
    ply_data = {}
    
    # Positions
    ply_data['x'] = means[:, 0].astype(np.float32)
    ply_data['y'] = means[:, 1].astype(np.float32)
    ply_data['z'] = means[:, 2].astype(np.float32)
    
    # Normals (set to zero as in the original implementation)
    num_points = means.shape[0]
    ply_data['nx'] = np.zeros(num_points, dtype=np.float32)
    ply_data['ny'] = np.zeros(num_points, dtype=np.float32)
    ply_data['nz'] = np.zeros(num_points, dtype=np.float32)
    
    # DC spherical harmonics
    ply_data['f_dc_0'] = features_dc[:, 0].astype(np.float32)
    ply_data['f_dc_1'] = features_dc[:, 1].astype(np.float32)
    ply_data['f_dc_2'] = features_dc[:, 2].astype(np.float32)
    
    features_rest = features_rest[valid_mask]
    features_rest = features_rest.cpu().numpy() if hasattr(features_rest, 'cpu') else features_rest
    # Flatten and add each component
    sh_rest_flat = features_rest.reshape(num_points, -1)  # [N, (sh_bases-1)*3]
    for i in range(sh_rest_flat.shape[1]):
        ply_data[f'f_rest_{i}'] = sh_rest_flat[:, i].astype(np.float32)

    # # Rest spherical harmonics
    # if features_rest is not None:
    #     features_rest = features_rest[valid_mask]
    #     features_rest = features_rest.cpu().numpy() if hasattr(features_rest, 'cpu') else features_rest
    #     # Flatten and add each component
    #     sh_rest_flat = features_rest.reshape(num_points, -1)  # [N, (sh_bases-1)*3]
    #     for i in range(sh_rest_flat.shape[1]):
    #         ply_data[f'f_rest_{i}'] = sh_rest_flat[:, i].astype(np.float32)
    # else:
    #     # Add zeros for rest coefficients if not present
    #     for i in range(45):  # Standard 3rd degree SH has 45 rest coefficients
    #         ply_data[f'f_rest_{i}'] = np.zeros(num_points, dtype=np.float32)
    
    # Opacity
    ply_data['opacity'] = opacities.squeeze().astype(np.float32)
    
    # Scales
    ply_data['scale_0'] = scales[:, 0].astype(np.float32)
    ply_data['scale_1'] = scales[:, 1].astype(np.float32)
    ply_data['scale_2'] = scales[:, 2].astype(np.float32)
    
    # Rotations (quaternions)
    ply_data['rot_0'] = quats[:, 0].astype(np.float32)
    ply_data['rot_1'] = quats[:, 1].astype(np.float32)
    ply_data['rot_2'] = quats[:, 2].astype(np.float32)
    ply_data['rot_3'] = quats[:, 3].astype(np.float32)
    
    return ply_data

def save_ply(ply_data: Dict[str, np.ndarray], output_path: str):
    """Save PLY data to file using custom PLY writer."""
    
    num_points = len(ply_data['x'])
    
    # Create PLY header
    properties = []
    property_order = []
    
    # Position properties
    for prop in ['x', 'y', 'z']:
        properties.append(f"property float {prop}")
        property_order.append(prop)
    
    # Normal properties
    for prop in ['nx', 'ny', 'nz']:
        properties.append(f"property float {prop}")
        property_order.append(prop)
    
    # Spherical harmonics DC
    for prop in ['f_dc_0', 'f_dc_1', 'f_dc_2']:
        properties.append(f"property float {prop}")
        property_order.append(prop)
    
    # Spherical harmonics rest coefficients
    for i in range(45):  # 45 rest coefficients for 3rd degree SH
        prop = f'f_rest_{i}'
        if prop in ply_data:
            properties.append(f"property float {prop}")
            property_order.append(prop)
    
    # Opacity
    properties.append("property float opacity")
    property_order.append("opacity")
    
    # Scales
    for prop in ['scale_0', 'scale_1', 'scale_2']:
        properties.append(f"property float {prop}")
        property_order.append(prop)
    
    # Rotations
    for prop in ['rot_0', 'rot_1', 'rot_2', 'rot_3']:
        properties.append(f"property float {prop}")
        property_order.append(prop)
    
    # Write PLY file
    with open(output_path, 'wb') as f:
        # Write header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode('ascii'))
        for prop in properties:
            f.write(f"{prop}\n".encode('ascii'))
        f.write(b"end_header\n")
        
        # Write binary data
        for i in range(num_points):
            for prop_name in property_order:
                if prop_name in ply_data:
                    value = ply_data[prop_name][i]
                    f.write(np.array([value], dtype=np.float32).tobytes())
                else:
                    # Write zero for missing properties
                    f.write(np.array([0.0], dtype=np.float32).tobytes())
    
    logger.info(f"Successfully saved PLY file to {output_path}")
    logger.info(f"Point cloud contains {num_points} points with {len(property_order)} properties")

def main():
    parser = argparse.ArgumentParser(description="Convert DriveStudio PTH to standard Gaussian Splatting PLY")
    parser.add_argument("pth_path", type=str, help="Path to input PTH file")
    parser.add_argument("-o", "--output", type=str, default=None, 
                       help="Output PLY path (default: same as input with .ply extension)")
    parser.add_argument("-m", "--model", type=str, default=None,
                       help="Model name to extract (e.g., 'Background', 'RigidNodes'). If not specified, will use first available.")
    parser.add_argument("--opacity-threshold", type=float, default=0.005,
                       help="Opacity threshold for filtering gaussians (default: 0.005)")
    parser.add_argument("--normalize", action="store_true",
                       help="Normalize positions")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate input
    if not os.path.exists(args.pth_path):
        logger.error(f"Input file does not exist: {args.pth_path}")
        return 1
    
    # Determine output path
    if args.output is None:
        model_suffix = f"_{args.model}" if args.model else ""
        output_path = os.path.splitext(args.pth_path)[0] + f"{model_suffix}_converted.ply"
    else:
        output_path = args.output
    
    try:
        # Load Gaussian model
        logger.info(f"Loading Gaussian model from {args.pth_path}")
        gaussian_params = load_gaussian_model(args.pth_path, args.model)
        
        logger.info(f"Found parameters: {list(gaussian_params.keys())}")
        for key, value in gaussian_params.items():
            if hasattr(value, 'shape'):
                logger.info(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        
        # Convert to PLY format
        logger.info("Converting to PLY format")
        ply_data = convert_to_ply_format(
            gaussian_params, 
            opacity_threshold=args.opacity_threshold,
            normalize_positions=args.normalize
        )
        
        # Save PLY file
        logger.info(f"Saving PLY file to {output_path}")
        save_ply(ply_data, output_path)
        
        logger.info("Conversion completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
