# Configuration Files

This directory contains configuration files for different DriveStudio models and datasets. Below are the key configuration sections to pay attention to when modifying these files.

## Key Configuration Points

### 1. Trainer Type
Choose between single trainer (background-only) or multi-trainer:

```yaml
trainer:
  type: models.trainers.SingleTrainerBackground
  # or
  type: models.trainers.MultiTrainer
```

### 2. 2DGS-Specific Settings

#### Rendering
Enable 2DGS renderer:
```yaml
trainer:
  render:
    use_2dgs: true  # Flag to use 2DGS renderer
```

#### Losses
When using 2DGS backbone, configure these losses:

```yaml
losses:
  # 2DGS specific losses
  normal:
    w: 0.05          # Normal consistency loss weight
    start_iter: 7000 # Start normal loss after this iteration
  
  # distortion:
  #   w: 0.01          # Distortion regularization weight
  #   start_iter: 3000 # Start distortion loss after this iteration
```

**Note:** For now, we are not using the distortion regularization that is in the original 2DGS paper. The distortion loss is commented out.

### 3. Background Model Initialization

Tune your initialization points ratio here:

```yaml
model:
  Background:
    type: models.gaussians.SurfelGaussians
    init:
      from_lidar:
        num_samples: 800_000
        return_color: True
      near_randoms: 100_000
      far_randoms: 100_000
```

### 4. Regularization Settings

Configure shape and sparsity regularization:

```yaml
reg:
  sharp_shape_reg:
    w: 1.
    step_interval: 10
    max_gauss_ratio: 10.  # threshold of ratio of gaussian max to min scale before applying regularization loss from the PhysGaussian paper
  # sparse_reg:
  #   w: 0.002
```

**Note:** More regularization options beyond sharp_shape or sparse reg terms are available - check the model scripts for additional options.

### 5. Novel View Rendering

Configure trajectory types for novel view synthesis:

```yaml
render:
  render_novel: 
    traj_types:
      - front_center_interp # type of trajectory for novel view synthesis
      # - s_curve
    fps: 24 # frames per second for novel view rendering
```

### 6. Logging and Checkpointing

Control training monitoring and saving frequencies:

```yaml
logging:
  vis_freq: 2000 # how often to visualize training stats
  print_freq: 500 # how often to print training stats
  saveckpt_freq: 30000 # how often to save checkpoints
  save_seperate_video: True # whether to save separate videos for each scene
```

## Configuration Files Overview

- `omnire_extended_cam_2dgs_background_only.yaml` - 2DGS with background-only training
- `omnire_extended_cam_3dgs_background_only.yaml` - 3DGS with background-only training
besides `background_only` and `no_smpl`, everything this from original drivestudio repo
- `omnire_extended_cam.yaml` - Extended camera configuration with full training
- `streetgs.yaml` - StreetGS configuration
- `deformablegs.yaml` - DeformableGS configuration
- `pvg.yaml` - PVG configuration

Dataset-specific configurations are located in the `datasets/` subdirectory, organized by dataset type (waymo, nuscenes, pandaset, etc.) and number of cameras.