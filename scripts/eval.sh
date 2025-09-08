ckpt_path=outputs/pandaset001/omnire_extended_cam_3dgs_background_only/checkpoint_final.pth
python -m tools.eval --resume_from $ckpt_path --extract_mesh 