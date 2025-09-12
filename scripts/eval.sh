ckpt_path=outputs/final_demo/pandaset8_3d_full/checkpoint_final.pth
python -m tools.eval --resume_from $ckpt_path --render_video