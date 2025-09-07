from typing import Dict

import torch
import logging

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from models.gaussians import PeriodicVibrationGaussians, SurfelGaussians
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere

"""
Single Gaussian Model Trainer (Background-only)

Differences from SingleTrainer:
- Only instantiates and trains the `Background` gaussian model (misc classes like `Sky`, `Affine`, `CamPose`, `CamPosePerturb` are allowed).
- During initialization from LiDAR, optionally filters out dynamic-object points using dataset instance seeds if object configs (RigidNodes/DeformableNodes/SMPLNodes) are present in the config, without creating their models.
- During loss computation, if `dynamic_masks` is provided in `image_infos`, it is fused into `egocar_masks` to exclude dynamic regions from losses (RGB/SSIM/Depth/Sky opacity) without touching global logic.
"""

logger = logging.getLogger()


class SingleTrainerBackground(BasicTrainer):
    def __init__(
        self, 
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)

        self._is_2dgs_model()

    def _is_2dgs_model(self):
        """
        Detect if any model is using 2DGS (SurfelGaussian)
        """
        for model_name in self.gaussian_classes.keys():
            model_cfg = self.model_config.get(model_name, {})
            if model_cfg.get("type", "").endswith("SurfelGaussians"):
                logger.info(f"Detected 2DGS model: {model_name}")
                assert self.render_cfg["use_2dgs"], "the yaml has to have this argument set to True"
                assert "normal" in self.losses_dict
                assert "distortion" in self.losses_dict
                assert "densify_grad_thresh" in self.gaussian_ctrl_general_cfg
                logger.info("Configured rendering for 2DGS")
                break

    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes (background only)
        self.gaussian_classes["Background"] = GSModelType.Background
     
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)

            # Only instantiate recognized classes: Background (gaussian) or misc classes
            model = None
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
            elif class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)
            else:
                # Skip any non-background, non-misc entries (e.g., RigidNodes/SMPL/Deformable)
                logger.info(f"Ignoring non-background class in config: {class_name}")
                continue

            self.models[class_name] = model
        
        logger.info(f"Initialized models: {self.models.keys()}")

        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # Collect optional dynamic instance dictionaries for filtering if provided via config
        # We do not instantiate those models; we only use their init settings to gather instance seeds
        rigidnode_pts_dict, deformnode_pts_dict = {}, {}
        rigidnode_pts_dict = dataset.get_init_objects(
                    cur_node_type='RigidNodes',
        )
        deformnode_pts_dict = dataset.get_init_objects(
                    cur_node_type='DeformableNodes',
                    exclude_smpl=False # state this explicitly so we include smpl instances here since this is only for pruning
                )

        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict}

        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
                    if sampled_time is not None:
                        sampled_time = torch.cat([sampled_time, torch.zeros(valid_pts.shape[0], 1).to(self.device)], dim=0)

                # Filter out dynamic instances if we have gathered them
                if len(allnode_pts_dict.keys()) > 0:
                    processed_init_pts = dataset.filter_pts_in_boxes(
                            seed_pts=sampled_pts,
                            seed_colors=sampled_color,
                            valid_instances_dict=allnode_pts_dict
                    )
                    sampled_pts = processed_init_pts["pts"]
                    sampled_color = processed_init_pts["colors"]
                    # sampled_time remains unchanged; these seeds are still background
                
                if isinstance(model, PeriodicVibrationGaussians):
                    model.create_from_pcd(
                        init_means=sampled_pts, init_colors=sampled_color, init_times=sampled_time
                    )
                else:
                    model.create_from_pcd(
                        init_means=sampled_pts, init_colors=sampled_color
                    )
                
            logger.info(f"Initialized {class_name} gaussians")
        logger.info(f"Initialized gaussians from pcd (background-only, dynamic filtered)")
        
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """
        
        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set
                
        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        
        return outputs
        
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # If dynamic_masks are provided, fuse them into egocar_masks so base losses ignore dynamic regions
        assert "dynamic_masks" in image_infos, "dynamic_masks are required for background-only trainer"
        dyn = image_infos["dynamic_masks"].float()
        if "egocar_masks" in image_infos:
            fused = torch.clamp(image_infos["egocar_masks"].float() + dyn, 0.0, 1.0)
        else:
            fused = dyn
        masked_infos = dict(image_infos)
        masked_infos["egocar_masks"] = fused
        # Optional: also zero-out lidar depth where dynamic to avoid incidental hits
        if "lidar_depth_map" in masked_infos:
            masked_infos["lidar_depth_map"] = masked_infos["lidar_depth_map"] * (1.0 - dyn)
        return super().compute_losses(outputs, masked_infos, cam_infos)


