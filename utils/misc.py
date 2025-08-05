# Miscellaneous utility functions for exporting point clouds.
import importlib
import logging
import os

import numpy as np
import open3d as o3d
import torch
import torch.distributed as dist

logger = logging.getLogger()

def import_str(string: str):
    """ Import a python module given string paths

    Args:
        string (str): The given paths

    Returns:
        Any: Imported python module / object
    """
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def export_points_to_ply(
    positions: torch.tensor,
    colors: torch.tensor,
    save_path: str,
    normalize: bool = False,
    ):
    # normalize points
    if normalize:
        aabb_min = positions.min(0)[0]
        aabb_max = positions.max(0)[0]
        positions = (positions - aabb_min) / (aabb_max - aabb_min)
    if isinstance(colors, torch.Tensor):
        positions = positions.cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()
    
    # clamp colors
    colors = np.clip(colors, a_min=0., a_max=1.)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)

def export_gaussians_to_ply(model, path, name='point_cloud.ply', aabb=None):
    model.eval()
    filename = os.path.join(path, name)
    map_to_tensors = {}
    
    with torch.no_grad():
        # Handle different model types - use _means for VanillaGaussians
        if hasattr(model, '_means'):
            positions = model._means
        elif hasattr(model, 'means'):
            positions = model.means
        else:
            raise AttributeError("Model must have either '_means' or 'means' attribute")
            
        if aabb is not None:
            aabb = aabb.to(positions.device)
            aabb_min, aabb_max = aabb[:3], aabb[3:]
            aabb_center = (aabb_min + aabb_max) / 2
            aabb_sacle_max = (aabb_max - aabb_min).max() / 2 * 1.1
            vis_mask = torch.logical_and(positions >= aabb_min, positions < aabb_max).all(-1)
        else:
            aabb_center = positions.mean(0)
            aabb_sacle_max = (positions - aabb_center).abs().max() * 1.1
            vis_mask = torch.ones_like(positions[:, 0], dtype=torch.bool)
            
        positions = ((positions[vis_mask] - aabb_center) / aabb_sacle_max).cpu().numpy()
        map_to_tensors["positions"] = o3d.core.Tensor(positions, o3d.core.float32)
        map_to_tensors["normals"] = o3d.core.Tensor(np.zeros_like(positions), o3d.core.float32)

        # Handle colors - use model.colors property for VanillaGaussians
        colors = model.colors[vis_mask].data.cpu().numpy()
        map_to_tensors["colors"] = (colors * 255).astype(np.uint8)
        for i in range(colors.shape[1]):
            map_to_tensors[f"f_dc_{i}"] = o3d.core.Tensor(colors[:, i].reshape(-1, 1), o3d.core.float32)

        # Handle SH features - VanillaGaussians uses shs_rest property  
        if hasattr(model, 'shs_rest'):
            shs = model.shs_rest[vis_mask].data.cpu().numpy()
        elif hasattr(model, '_features_rest'):
            shs = model._features_rest[vis_mask].data.cpu().numpy()
        else:
            shs = np.zeros((colors.shape[0], 0, 3))
            
        # Check sh_degree from different possible locations
        sh_degree = 0
        if hasattr(model, 'sh_degree'):
            sh_degree = model.sh_degree
        elif hasattr(model, 'config') and hasattr(model.config, 'sh_degree'):
            sh_degree = model.config.sh_degree
        elif hasattr(model, 'ctrl_cfg') and hasattr(model.ctrl_cfg, 'sh_degree'):
            sh_degree = model.ctrl_cfg.sh_degree
            
        if sh_degree > 0 and shs.shape[1] > 0:
            shs = shs.reshape((colors.shape[0], -1, 3))
            for i in range(shs.shape[1]):
                for j in range(3):
                    map_to_tensors[f"f_rest_{i*3+j}"] = o3d.core.Tensor(shs[:, i, j].reshape(-1, 1), o3d.core.float32)

        # Handle opacities
        if hasattr(model, 'get_opacity'):
            opacities = model.get_opacity[vis_mask].data.cpu().numpy()
        elif hasattr(model, 'opacities'):
            opacities = model.opacities[vis_mask].data.cpu().numpy()
        else:
            opacities = model._opacities[vis_mask].data.cpu().numpy()
        map_to_tensors["opacity"] = o3d.core.Tensor(opacities.reshape(-1, 1), o3d.core.float32)

        # Handle scales
        if hasattr(model, 'get_scaling'):
            scales = model.get_scaling[vis_mask].data.cpu().numpy()
        elif hasattr(model, 'scales'):
            scales = model.scales[vis_mask].data.cpu().numpy()
        else:
            scales = torch.exp(model._scales[vis_mask]).data.cpu().numpy()
        
        for i in range(scales.shape[1]):
            map_to_tensors[f"scale_{i}"] = o3d.core.Tensor(scales[:, i].reshape(-1, 1), o3d.core.float32)

        # Handle quaternions
        if hasattr(model, 'get_quats'):
            quats = model.get_quats[vis_mask].data.cpu().numpy()
        elif hasattr(model, 'quats'):
            quats = model.quats[vis_mask].data.cpu().numpy()
        else:
            quats = model._quats[vis_mask].data.cpu().numpy()
            quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)  # normalize

        for i in range(4):
            map_to_tensors[f"rot_{i}"] = o3d.core.Tensor(quats[:, i].reshape(-1, 1), o3d.core.float32)

    pcd = o3d.t.geometry.PointCloud(map_to_tensors)
    o3d.t.io.write_point_cloud(str(filename), pcd)
    
    logger.info(f"Exported point cloud to {filename}, containing {vis_mask.sum().item()} points.")

def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0


def get_world_size():
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0