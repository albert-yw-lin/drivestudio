"""
Filename: surfel.py
Location: models/gaussians/surfel.py

Author: Implementation for 2D Gaussian Splatting (2DGS) in DriveStudio

Description:
Implementation of 2D Gaussian Splatting based on the paper:
"2D Gaussian Splatting for Geometrically Accurate Radiance Fields"
by Huang et al. (SIGGRAPH 2024)

This model represents the scene using 2D oriented disks (surfels) instead of 3D ellipsoids.
"""

from typing import Dict, List, Tuple, Optional
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np

from models.gaussians.vanilla import VanillaGaussians
from models.gaussians.basics import *

logger = logging.getLogger()


class SurfelGaussians(VanillaGaussians):
    """
    2D Gaussian Splatting model using oriented disks (surfels) for scene representation.
    
    Key differences from 3DGS:
    - Uses 2D scales (major and minor axis of the disk)
    - Represents surfaces with oriented disks
    - Includes surface normal computation
    - Supports depth regularization strategies
    """
    
    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        """
        Initialize 2D Gaussian Splatting model.
        """

        # Ensure 2D Gaussian mode is enabled for 2DGS
        ctrl.gaussian_2d = True
        
        super().__init__(
            class_name=class_name,
            ctrl=ctrl,
            reg=reg,
            networks=networks,
            scene_scale=scene_scale,
            scene_origin=scene_origin,
            num_train_images=num_train_images,
            device=device,
            **kwargs
        )

        logger.info(f"Initialized 2D Gaussian Splatting model")

    @property
    def get_scaling(self):
        """
        Override to ensure proper 2D disk representation.
        For 2DGS, the third dimension (normal direction) is set to near-zero.
        """
        # Get 2D scales (exponential of log scales)
        scales_2d = torch.exp(self._scales)  # [N, 2]
        
        # Create 3D scales with near-zero thickness in normal direction
        # This creates flat disks rather than ellipsoids
        thickness = torch.full((scales_2d.shape[0], 1), 1e-8, device=self.device)
        scales_3d = torch.cat([scales_2d, thickness], dim=-1)  # [N, 3]
        
        return scales_3d
    
    @property
    def get_scaling_2d(self):
        """
        Get 2D scaling parameters (major and minor axis of the disk).
        """
        return torch.exp(self._scales)  # [N, 2]
    
    @property
    def get_scaling_2d_raw(self):
        """
        Get 2D scaling parameters (major and minor axis of the disk).
        """
        return self._scales  # [N, 2]
    
    def compute_reg_loss(self):
        """
        Compute regularization losses including 2DGS-specific losses.
        """
        # TODO: check if every reg loss in super is needed for 2dgs
        # Get base regularization losses from parent
        # loss_dict = super().compute_reg_loss() #NOTE: not use 3dgs's reg for now 
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling_2d
                # scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                # Add epsilon to prevent division by zero and ensure numerical stability
                scale_min = torch.clamp(scale_exp.amin(dim=-1), min=1e-6)
                scale_max = scale_exp.amax(dim=-1)
                scale_ratio = scale_max / scale_min
                scale_reg = torch.maximum(scale_ratio, torch.tensor(max_gauss_ratio, device=scale_exp.device)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg
        
        # the same as vanilla's 
        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        
        return loss_dict
    
    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        Override to handle 2D scales properly during splitting.
        """
        # For 2D Gaussians, we need to be careful with the scaling
        n_splits = split_mask.sum().item()
        print(f"    Split (2D): {n_splits}")
        
        # Sample in 2D disk plane, add zero for z
        centered_samples_2d = torch.randn((samps * n_splits, 2), device=self.device)
        centered_samples = torch.cat([
            centered_samples_2d,
            torch.zeros((samps * n_splits, 1), device=self.device)
        ], dim=-1)
        
        # Scale samples by 2D scales
        scales_3d = self.get_scaling[split_mask]  # This already has near-zero z
        scaled_samples = scales_3d.repeat(samps, 1) * centered_samples
        
        # Rotate samples
        quats = self.quat_act(self._quats[split_mask])
        rots = quat_to_rotmat(quats.repeat(samps, 1))
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        
        # Copy features
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        
        # Scale down the 2D scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        
        # Copy rotations
        new_quats = self._quats[split_mask].repeat(samps, 1)
        
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats