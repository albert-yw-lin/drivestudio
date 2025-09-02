export PYTHONPATH=$(pwd)

# Check if we're in a conda/mamba environment and nvcc exists there
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
    NVCC_PATH="$CONDA_PREFIX/bin/nvcc"
    CUDA_PATH="$CONDA_PREFIX"
    echo "Using CUDA from mamba environment: $CUDA_PATH"
else
    # Fallback to system nvcc
    NVCC_PATH=$(which nvcc)
    CUDA_PATH=$(dirname $(dirname $NVCC_PATH))
    echo "Using system CUDA: $CUDA_PATH"
fi

export CUDA_PATH=$CUDA_PATH
export CUDA_HOME=$CUDA_PATH

# export CC=gcc-11
# export CXX=g++-11

# export LD_LIBRARY_PATH=$CUDA_HOME/lib/python3.9/site-packages/torch/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# Keep LD paths minimal and avoid leaking other envs' Torch libs
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/lib/x86_64-linux-gnu
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export TF_FORCE_GPU_ALLOW_GROWTH=true