import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SimpleCam:
    """Simple camera for mesh extraction"""
    camtoworlds: torch.Tensor
    Ks: torch.Tensor
    H: int
    W: int

class BackgroundMeshExtractor:
    """Mesh extractor specifically for Background gaussians in MultiTrainer"""
    
    def __init__(self, scene_scale: float = 30.0):
        self.scene_scale = scene_scale
        self.depthmaps = []
        self.rgbmaps = []
        self.cameras = []
        
    @torch.no_grad()
    def extract_background_depths_from_trainer(
        self, 
        trainer,
        dataset,
        num_views: int = None,
        camera_downscale: int = 2
    ):
        """Extract depth maps specifically from Background gaussians"""
        self.depthmaps = []
        self.rgbmaps = []
        self.cameras = []
        
        # Ensure we render each class separately
        original_render_each_class = trainer.render_each_class
        trainer.render_each_class = True
        trainer.set_eval()

        # Sample views evenly from the dataset
        total_views = len(dataset.full_image_set)
        step = max(1, total_views // num_views) if num_views is not None else 1
        sample_indices = list(range(0, total_views, step))[:num_views]
        
        for idx in tqdm(sample_indices, desc="Extracting Background depths"):
            image_infos, cam_infos = dataset.full_image_set.get_image(
                idx, camera_downscale
            )
            
            # Move to GPU
            for k, v in image_infos.items():
                if isinstance(v, torch.Tensor):
                    image_infos[k] = v.cuda()
            for k, v in cam_infos.items():
                if isinstance(v, torch.Tensor):
                    cam_infos[k] = v.cuda()
            
            # Render - this will include Background_depth and Background_rgb
            outputs = trainer(image_infos, cam_infos, novel_view=True)
            
            # Check if Background model exists and was rendered
            if "Background_depth" not in outputs:
                raise ValueError("Background_depth not found in outputs. "
                               "Ensure Background model exists in trainer.")
            
            # Store Background-only results
            self.depthmaps.append(outputs["Background_depth"].squeeze().cpu())
            
            # For RGB, we might want Background + Sky
            if "Background_rgb" in outputs:
                # Get background RGB
                bg_rgb = outputs["Background_rgb"]
                
                # Optionally blend with sky for empty regions
                if "rgb_sky" in outputs and "Background_opacity" in outputs:
                    bg_opacity = outputs["Background_opacity"]
                    sky_rgb = outputs["rgb_sky"]
                    # Blend: bg + sky * (1 - opacity)
                    blended_rgb = bg_rgb + sky_rgb * (1.0 - bg_opacity)
                    self.rgbmaps.append(blended_rgb.cpu())
                else:
                    self.rgbmaps.append(bg_rgb.cpu())
            else:
                # Fallback to combined rgb if Background_rgb not available
                self.rgbmaps.append(outputs["rgb"].cpu())
            
            # Store camera
            cam = SimpleCam(
                camtoworlds=cam_infos["camera_to_world"],
                Ks=cam_infos["intrinsics"],
                H=cam_infos["height"].item() if torch.is_tensor(cam_infos["height"]) else cam_infos["height"],
                W=cam_infos["width"].item() if torch.is_tensor(cam_infos["width"]) else cam_infos["width"]
            )
            self.cameras.append(cam)
        
        # Restore original setting
        trainer.render_each_class = original_render_each_class
        
        # Estimate bounding sphere for TSDF
        self.estimate_bounding_sphere()
        
        print(f"Extracted {len(self.depthmaps)} depth maps from Background model")
    
    def estimate_bounding_sphere(self):
        """Estimate scene bounds from camera positions"""
        cam_positions = torch.stack([
            cam.camtoworlds[:3, 3] for cam in self.cameras
        ])
        center = cam_positions.mean(dim=0)
        radius = (cam_positions - center).norm(dim=-1).max()
        self.radius = radius.item()
        self.center = center
        
    def to_open3d_camera(self, cam: SimpleCam):
        """Convert camera to Open3D format"""
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=cam.W,
            height=cam.H,
            fx=cam.Ks[0, 0].item(),
            fy=cam.Ks[1, 1].item(),
            cx=cam.Ks[0, 2].item(),
            cy=cam.Ks[1, 2].item()
        )
        
        # Convert camtoworld to world2cam for Open3D
        world2cam = torch.linalg.inv(cam.camtoworlds)
        extrinsic = world2cam.cpu().numpy()
        
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        return camera
    
    @torch.no_grad()
    def extract_mesh_tsdf(
        self,
        voxel_size: float = 0.05, # emperically tuned number
        sdf_trunc: float = None,
        depth_trunc: float = None,
        depth_threshold: float = None
    ):
        """Extract mesh using TSDF fusion
        
        Args:
            voxel_size: Voxel size for TSDF volume
            sdf_trunc: Truncation distance for SDF
            depth_trunc: Maximum depth range for integration
            depth_threshold: Optional minimum depth threshold to filter out sky
        """
        
        # Auto-compute parameters if not provided
        # breakpoint()
        # if voxel_size is None:
        #     voxel_size = self.radius * 2 / 512
        if sdf_trunc is None:
            sdf_trunc = voxel_size * 5
        if depth_trunc is None:
            depth_trunc = self.radius*2
            
        print(f"TSDF Fusion Parameters:")
        print(f"  voxel_size: {voxel_size:.4f}")
        print(f"  sdf_trunc: {sdf_trunc:.4f}")
        print(f"  depth_trunc: {depth_trunc:.2f}")
        print(f"  scene_radius: {self.radius:.2f}")
        
        # Create TSDF volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Integrate all views
        for i, (depth, rgb, cam) in enumerate(tqdm(
            zip(self.depthmaps, self.rgbmaps, self.cameras),
            desc="TSDF integration", total=len(self.depthmaps)
        )):
            cam_o3d = self.to_open3d_camera(cam)
            
            # Optional: threshold depth to remove far away points (sky)
            if depth_threshold is not None:
                depth_np = depth.numpy()
                depth_np[depth_np > depth_threshold] = 0
                depth = torch.from_numpy(depth_np)
            # breakpoint()
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(np.clip(rgb.numpy(), 0, 1) * 255, 
                              dtype=np.uint8, order="C")
                ),
                o3d.geometry.Image(
                    np.asarray(depth.unsqueeze(-1).numpy(), order="C")
                ),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0
            )
            
            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, 
                           extrinsic=cam_o3d.extrinsic)
        
        # Extract mesh
        mesh = volume.extract_triangle_mesh()
        print(f"Extracted mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        return mesh
    
    def post_process_mesh(self, mesh, min_cluster_size: int = 50):
        """Remove small disconnected components"""
        import copy
        mesh_clean = copy.deepcopy(mesh)
        
        # Cluster connected triangles
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            triangle_clusters, cluster_n_triangles, _ = (
                mesh_clean.cluster_connected_triangles()
            )
        
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Remove small clusters
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_cluster_size
        mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        mesh_clean.remove_unreferenced_vertices()
        mesh_clean.remove_degenerate_triangles()
        
        print(f"Post-processing: {len(mesh.vertices)} -> {len(mesh_clean.vertices)} vertices")
        return mesh_clean