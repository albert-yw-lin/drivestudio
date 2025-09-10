ckpt_path=outputs/final_demo/nuscenes3_3d_full/checkpoint_final.pth
python -m tools.eval --resume_from $ckpt_path --extract_mesh --mesh_voxel_size 0.1 --export_ply