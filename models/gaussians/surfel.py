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
        
        # 2DGS specific parameters
        self.depth_ratio = self.ctrl_cfg.get("depth_ratio", 0.0)  # 0 for mean, 1 for median depth
        
        # Surface normal for each Gaussian (computed from rotation quaternion)
        self._normals = None
        
        logger.info(f"Initialized 2D Gaussian Splatting model with depth_ratio={self.depth_ratio}")
    
    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        """
        Initialize 2D Gaussians from point cloud.
        Override parent to ensure 2D scaling is properly initialized.
        """
        # Call parent implementation
        super().create_from_pcd(init_means, init_colors)
        
        # Compute initial normals from quaternions
        self._update_normals()
        
        logger.info(f"Created {self.num_points} 2D Gaussians from point cloud")
    
    def _update_normals(self):
        """
        Update surface normals from quaternion rotations.
        The normal is the z-axis of the rotation matrix derived from the quaternion.
        """
        # Convert quaternions to rotation matrices
        quats_normalized = self.quat_act(self._quats)
        R = quat_to_rotmat(quats_normalized)  # [N, 3, 3]
        
        # Extract the z-axis (third column) as the normal direction
        # This represents the orientation of the disk
        self._normals = R[:, :, 2]  # [N, 3]
        
        # Normalize to ensure unit normals
        self._normals = self._normals / (torch.norm(self._normals, dim=-1, keepdim=True) + 1e-8)
    
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

    
    @property
    def get_normals(self):
        """
        Get surface normals for each Gaussian.
        """
        if self._normals is None or self._normals.shape[0] != self.num_points:
            self._update_normals()
        return self._normals
    
    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        """
        Override to include normals in the gaussian dictionary for 2DGS rendering.
        """
        # Get base gaussians from parent
        gs_dict = super().get_gaussians(cam)
        
        self._update_normals()
        gs_dict['_normals'] = self._normals[self.filter_mask]
        
        return gs_dict
    
    def compute_reg_loss(self):
        """
        Compute regularization losses including 2DGS-specific losses.
        """
        # TODO: check if every reg loss in super is needed for 2dgs
        # Get base regularization losses from parent
        loss_dict = super().compute_reg_loss()
            
        # Normal consistency loss (will be computed in trainer with rendered normals)
        # Just placeholder here - actual computation needs rendered outputs
        
        # Depth distortion regularization (will be computed in trainer)
        # Just placeholder here - actual computation needs rendered outputs
        
        # Add scale anisotropy regularization for 2D Gaussians
        # Encourages disks to be more circular rather than extremely elongated
        if hasattr(self.reg_cfg, 'anisotropy_reg') and self.reg_cfg.anisotropy_reg is not None:
            anisotropy_reg = self.reg_cfg.anisotropy_reg
            scales_2d = self.get_scaling_2d  # [N, 2]
            scale_ratio = scales_2d.max(dim=-1).values / (scales_2d.min(dim=-1).values + 1e-8)
            max_ratio = anisotropy_reg.get("max_ratio", 10.0)
            anisotropy_loss = torch.clamp(scale_ratio - max_ratio, min=0.0).mean()
            loss_dict["anisotropy_reg"] = anisotropy_loss * anisotropy_reg.w
        
        return loss_dict
    
    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        """
        Override refinement to update normals after densification operations.
        """
        # Call parent's refinement logic
        super().refinement_after(step, optimizer)
        
        # Update normals after any changes to gaussians
        self._update_normals()
    
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
    
    # def export_gaussians_to_ply(self, alpha_thresh: float) -> Dict:
    #     """
    #     Export gaussians with additional 2DGS-specific information.
    #     """
    #     # Get base export from parent
    #     export_dict = super().export_gaussians_to_ply(alpha_thresh)
        
    #     activated_opacities = self.get_opacity
    #     mask = activated_opacities.squeeze() > alpha_thresh
        
    #     # Add normals
    #     self._update_normals()
    #     export_dict["normals"] = self._normals[mask]
        
    #     # Add 2D scales
    #     export_dict["scales_2d"] = self.get_scaling_2d[mask]
        
    #     # Add quaternions
    #     export_dict["quaternions"] = self.quat_act(self._quats[mask])
        
    #     return export_dict
    
    def compute_2dgs_regularization(self, rendered_outputs: Dict, viewpoint_camera) -> Dict:
        """
        Compute 2DGS-specific regularization losses that require rendered outputs.
        This should be called from the trainer after rendering.
        
        Args:
            rendered_outputs: Dictionary containing rendered outputs
            viewpoint_camera: Camera object
            
        Returns:
            Dictionary of regularization losses
        """
        losses = {}
        
        # Normal consistency loss
        lambda_normal = self.reg_cfg.get("lambda_normal", 0.0)
        if lambda_normal > 0 and 'normal' in rendered_outputs:
            rendered_normal = rendered_outputs['normal']
            # Compute pseudo surface normal from depth
            if 'depth' in rendered_outputs:
                depth = rendered_outputs['depth']
                pseudo_normal = self._depth_to_normal(depth, viewpoint_camera)
                
                # Compute cosine similarity between rendered and pseudo normals
                cos_sim = torch.nn.functional.cosine_similarity(
                    rendered_normal.reshape(-1, 3),
                    pseudo_normal.reshape(-1, 3),
                    dim=-1
                )
                normal_loss = (1.0 - cos_sim).mean()
                losses['normal_consistency'] = lambda_normal * normal_loss
        
        # Depth distortion loss
        lambda_distortion = self.reg_cfg.get("lambda_distortion", 0.0)
        if lambda_distortion > 0 and 'depth_distortion' in rendered_outputs:
            depth_distortion = rendered_outputs['depth_distortion']
            losses['depth_distortion'] = lambda_distortion * depth_distortion.mean()
        
        return losses
    
    def _depth_to_normal(self, depth: torch.Tensor, viewpoint_camera) -> torch.Tensor:
        """
        Convert depth map to normal map using finite differences.
        
        Args:
            depth: Depth map [H, W, 1] or [1, H, W]
            viewpoint_camera: Camera object
            
        Returns:
            Normal map [H, W, 3] or [3, H, W]
        """
        if depth.dim() == 3 and depth.shape[0] == 1:
            depth = depth.squeeze(0)  # [H, W]
        elif depth.dim() == 3 and depth.shape[2] == 1:
            depth = depth.squeeze(2)  # [H, W]
        
        H, W = depth.shape
        
        # Compute gradients using finite differences
        # Pad depth to handle boundaries
        depth_pad = torch.nn.functional.pad(depth, (1, 1, 1, 1), mode='replicate')
        
        # Central differences
        dz_dx = (depth_pad[1:-1, 2:] - depth_pad[1:-1, :-2]) / 2.0
        dz_dy = (depth_pad[2:, 1:-1] - depth_pad[:-2, 1:-1]) / 2.0
        
        # Get camera intrinsics
        fx = viewpoint_camera.fx if hasattr(viewpoint_camera, 'fx') else W / (2 * torch.tan(viewpoint_camera.FoVx / 2))
        fy = viewpoint_camera.fy if hasattr(viewpoint_camera, 'fy') else H / (2 * torch.tan(viewpoint_camera.FoVy / 2))
        
        # Compute normal from gradients
        # n = normalize([-dz/dx * z/fx, -dz/dy * z/fy, 1])
        normal_x = -dz_dx * depth / fx
        normal_y = -dz_dy * depth / fy
        normal_z = torch.ones_like(depth)
        
        # Stack and normalize
        normals = torch.stack([normal_x, normal_y, normal_z], dim=-1)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        
        return normals