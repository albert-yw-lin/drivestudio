#!/bin/bash

# eval_for_demo.sh - Evaluate all final checkpoints in the final_demo directory
# This script runs evaluation for every model in outputs/final_demo/

echo "Starting evaluation for all final demo checkpoints..."
echo "=============================================="

# Base directory containing all the demo outputs
BASE_DIR="outputs/final_demo"

# Function to evaluate a single checkpoint
evaluate_checkpoint() {
    local checkpoint_path="$1"
    local model_name="$2"
    
    echo ""
    echo "Evaluating: $model_name"
    echo "Checkpoint: $checkpoint_path"
    echo "----------------------------------------"
    
    # Run the evaluation
    python -m tools.eval --resume_from "$checkpoint_path" --render_video
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated: $model_name"
    else
        echo "✗ Failed to evaluate: $model_name"
    fi
}

# Find and evaluate all final checkpoints
for model_dir in "$BASE_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        checkpoint_path="$model_dir/checkpoint_final.pth"
        
        # Check if the final checkpoint exists
        if [ -f "$checkpoint_path" ]; then
            evaluate_checkpoint "$checkpoint_path" "$model_name"
        else
            echo "⚠ Warning: Final checkpoint not found for $model_name"
        fi
    fi
done

echo ""
echo "=============================================="
echo "Evaluation completed for all final demo checkpoints!"
