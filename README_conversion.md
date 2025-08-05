# Gaussian Splatting Model Conversion Tools

This directory contains tools to convert and inspect Gaussian Splatting models between DriveStudio's PTH format and the standard PLY format.

## Files

- `convert_pth_to_ply.py` - Convert PTH models to PLY format
- `inspect_gaussian_files.py` - Inspect PTH and PLY files to understand their structure

## Format Comparison

### PTH Format (DriveStudio)
Contains PyTorch parameters stored in checkpoints:
- `_means`: [N, 3] - Gaussian centers  
- `_scales`: [N, 3] - Gaussian scales (log space)
- `_quats`: [N, 4] - Quaternion rotations
- `_features_dc`: [N, 3] - DC component of spherical harmonics
- `_features_rest`: [N, (sh_degree+1)^2-1, 3] - Rest of spherical harmonics
- `_opacities`: [N, 1] - Gaussian opacities (logit space)
- Additional temporal parameters for PVG models: `_taus`, `_betas`, `_velocity`

### PLY Format (Standard Gaussian Splatting)
Contains 62 properties per vertex:
- `x, y, z`: Positions
- `nx, ny, nz`: Normals (usually zeros)
- `f_dc_0, f_dc_1, f_dc_2`: DC spherical harmonics
- `f_rest_0` to `f_rest_44`: Rest spherical harmonics (45 coefficients)
- `opacity`: Gaussian opacity
- `scale_0, scale_1, scale_2`: Gaussian scales
- `rot_0, rot_1, rot_2, rot_3`: Quaternion rotations

## Usage Examples

### Inspect Files
```bash
# Inspect a PTH checkpoint file
python inspect_gaussian_files.py checkpoint_final.pth

# Inspect a PLY file  
python inspect_gaussian_files.py point_cloud.ply

# Inspect multiple files
python inspect_gaussian_files.py checkpoint.pth point_cloud.ply
```

### Convert PTH to PLY
```bash
# Convert Background model (will auto-detect available models)
python convert_pth_to_ply.py checkpoint_final.pth -m Background -v

# Convert RigidNodes model
python convert_pth_to_ply.py checkpoint_final.pth -m RigidNodes -v

# Convert with custom output path
python convert_pth_to_ply.py checkpoint_final.pth -m Background -o my_gaussians.ply

# Convert with different opacity threshold
python convert_pth_to_ply.py checkpoint_final.pth -m Background --opacity-threshold 0.01

# Convert without normalizing positions (keep original coordinates)
python convert_pth_to_ply.py checkpoint_final.pth -m Background --no-normalize
```

### Options
- `-m, --model`: Specify which model to extract (Background, RigidNodes, etc.)
- `-o, --output`: Custom output PLY file path
- `--opacity-threshold`: Filter out gaussians below this opacity (default: 0.005)
- `--no-normalize`: Don't normalize positions to [-1, 1] range
- `-v, --verbose`: Verbose output

## Technical Notes

1. **Coordinate Systems**: The conversion preserves the original coordinate system by default but offers normalization as an option.

2. **Opacity Filtering**: Low-opacity gaussians are filtered out to reduce file size. Adjust `--opacity-threshold` as needed.

3. **Spherical Harmonics**: The conversion handles different SH degrees automatically. Missing coefficients are padded with zeros.

4. **Multiple Models**: DriveStudio checkpoints often contain multiple Gaussian models (Background, RigidNodes, SMPLNodes, etc.). Use the `-m` flag to specify which one to convert.

5. **Memory Usage**: Large models (>1M gaussians) may require significant RAM during conversion.

## Example: Converting from a DriveStudio Checkpoint

```bash
# First, inspect to see available models
python inspect_gaussian_files.py checkpoint_final.pth

# Convert the background model
python convert_pth_to_ply.py checkpoint_final.pth -m Background -v

# The output will be: checkpoint_final_Background_converted.ply
```

This creates a standard Gaussian Splatting PLY file that can be loaded in viewers like the original Gaussian Splatting codebase or other compatible tools.
