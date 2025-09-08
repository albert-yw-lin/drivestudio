ckpt_path=outputs/pandaset001/omnire_extended_cam_2dgs_background_only_no_distortion/checkpoint_final.pth
python -m tools.eval --resume_from $ckpt_path --extract_mesh --mesh_voxel_size 0.05