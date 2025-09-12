config_file=configs/paper_legacy/omnire.yaml
output_root=outputs
project=nuscenes1
expname=omnire_paper_legacy
dataset=nuscenes/6cams
scene_idx=1
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame


python -m tools.train \
    --config_file $config_file \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=$dataset \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep