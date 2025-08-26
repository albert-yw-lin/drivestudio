ckpt_path=outputs/nuscenes1/omnire_2dgs/checkpoint_final.pth
python -m tools.eval --resume_from $ckpt_path --export_ply