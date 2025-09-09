#!/bin/bash

# Function to run training with error handling
run_training() {
    local config_file=$1
    local expname=$2
    local dataset=$3
    local scene_idx=$4
    local start_timestep=$5
    local end_timestep=$6
    local output_root=$7
    local project=$8
    
    echo "=========================================="
    echo "Starting training: $expname"
    echo "Config: $config_file"
    echo "Dataset: $dataset, Scene: $scene_idx"
    echo "=========================================="
    
    if python -m tools.train \
        --config_file "$config_file" \
        --output_root "$output_root" \
        --project "$project" \
        --run_name "$expname" \
        dataset="$dataset" \
        data.scene_idx="$scene_idx" \
        data.start_timestep="$start_timestep" \
        data.end_timestep="$end_timestep"; then
        echo "✓ SUCCESS: $expname completed successfully"
    else
        echo "✗ ERROR: $expname failed with exit code $?"
        echo "Continuing with next training..."
    fi
    echo ""
}

output_root=outputs
project=final_demo
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

# waymo300
dataset=waymo/5cams_no_smpl
scene_idx=300

config_file=configs/omnire_extended_cam_2dgs_background_only.yaml
expname=waymo300_2d
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"

config_file=configs/omnire_extended_cam_3dgs_background_only.yaml
expname=waymo300_3d
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"

config_file=configs/omnire_extended_cam_no_smpl.yaml
expname=waymo300_3d_full
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"




# waymo552
dataset=waymo/5cams
scene_idx=552

config_file=configs/omnire_extended_cam_2dgs_background_only.yaml
expname=waymo552_2d
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"

config_file=configs/omnire_extended_cam_3dgs_background_only.yaml
expname=waymo552_3d
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"

config_file=configs/omnire_extended_cam.yaml
expname=waymo552_3d_full
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"




# waymo250
dataset=waymo/5cams_no_smpl
scene_idx=250

config_file=configs/omnire_extended_cam_2dgs_background_only.yaml
expname=waymo250_2d
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"

config_file=configs/omnire_extended_cam_3dgs_background_only.yaml
expname=waymo250_3d
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"

config_file=configs/omnire_extended_cam_no_smpl.yaml
expname=waymo250_3d_full
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"




# waymo350
dataset=waymo/5cams_no_smpl
scene_idx=350

config_file=configs/omnire_extended_cam_2dgs_background_only.yaml
expname=waymo350_2d
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"

config_file=configs/omnire_extended_cam_3dgs_background_only.yaml
expname=waymo350_3d
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"

config_file=configs/omnire_extended_cam_no_smpl.yaml
expname=waymo350_3d_full
run_training "$config_file" "$expname" "$dataset" "$scene_idx" "$start_timestep" "$end_timestep" "$output_root" "$project"