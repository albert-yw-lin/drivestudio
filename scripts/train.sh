config_file=configs/omnire_extended_cam.yaml
output_root=outputs
project=nuscenes5
expname=omnire_full_ssim_0_4
dataset=nuscenes/6cams
scene_idx=5
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame


python -m tools.train \
    --config_file $config_file \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    --gpu_id 0 \
    dataset=$dataset \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep