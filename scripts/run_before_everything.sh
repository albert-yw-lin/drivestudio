export PYTHONPATH=$(pwd)
export CUDA_HOME=/root/miniforge3/envs/drivestudio #NOTE: You need to set this to your CUDA installation path
export LD_LIBRARY_PATH=$CUDA_HOME/lib/python3.9/site-packages/torch/lib:$CUDA_HOME/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH