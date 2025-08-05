#!/usr/bin/env python3
"""
Script to inspect Gaussian Splatting PTH files and PLY files to understand their structure.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

def inspect_pth_file(pth_path: str):
    """Inspect a PTH file and show its structure."""
    print(f"\n=== Inspecting PTH file: {pth_path} ===")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(pth_path, map_location='cpu')
        print(f"File size: {os.path.getsize(pth_path) / (1024*1024):.2f} MB")
        print(f"Top-level keys: {list(checkpoint.keys())}")
        
        # Look for different possible structures
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    print(f"\n{key} contains: {list(value.keys())}")
                    # Look for Gaussian parameters
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'shape'):
                            print(f"  {sub_key}: shape {sub_value.shape}, dtype {sub_value.dtype}")
                        elif isinstance(sub_value, dict):
                            print(f"  {sub_key}: dict with keys {list(sub_value.keys())}")
                elif hasattr(value, 'shape'):
                    print(f"{key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"{key}: {type(value)}")
        
        # Look specifically for Gaussian parameters
        gaussian_params = {}
        def find_gaussian_params(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if any(param in key for param in ['_means', '_scales', '_quats', '_features_dc', '_features_rest', '_opacities', '_taus', '_betas', '_velocity']):
                        gaussian_params[new_prefix] = value
                    elif isinstance(value, dict):
                        find_gaussian_params(value, new_prefix)
        
        find_gaussian_params(checkpoint)
        
        if gaussian_params:
            print(f"\n=== Found Gaussian Parameters ===")
            for key, value in gaussian_params.items():
                if hasattr(value, 'shape'):
                    print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print("\nNo Gaussian parameters found!")
            
    except Exception as e:
        print(f"Error loading PTH file: {e}")

def inspect_ply_file(ply_path: str):
    """Inspect a PLY file and show its structure."""
    print(f"\n=== Inspecting PLY file: {ply_path} ===")
    
    try:
        with open(ply_path, 'rb') as f:
            # Read header
            line = f.readline().decode('ascii').strip()
            if line != 'ply':
                print("Not a valid PLY file")
                return
                
            properties = []
            vertex_count = 0
            
            while True:
                line = f.readline().decode('ascii').strip()
                if line == 'end_header':
                    break
                elif line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('property'):
                    prop_parts = line.split()
                    prop_type = prop_parts[1]
                    prop_name = prop_parts[2]
                    properties.append((prop_name, prop_type))
            
            print(f"File size: {os.path.getsize(ply_path) / (1024*1024):.2f} MB")
            print(f"Vertex count: {vertex_count:,}")
            print(f"Properties ({len(properties)}):")
            for name, prop_type in properties:
                print(f"  {name}: {prop_type}")
                
    except Exception as e:
        print(f"Error reading PLY file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inspect PTH and PLY Gaussian Splatting files")
    parser.add_argument("files", nargs="+", help="Files to inspect")
    
    args = parser.parse_args()
    
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            continue
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pth':
            inspect_pth_file(file_path)
        elif ext == '.ply':
            inspect_ply_file(file_path)
        else:
            print(f"Unknown file type: {file_path}")

if __name__ == "__main__":
    main()
