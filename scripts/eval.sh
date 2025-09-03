ckpt_path=outputs/waymo552/omnire_extended_cam/checkpoint_final.pth
python -m tools.eval --resume_from $ckpt_path --extract_mesh