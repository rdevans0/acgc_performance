

MY_PATH=$(dirname $BASH_SOURCE)
export CUDA_HOME=$MY_PATH

echo "Added path $CUDA_HOME"


export CUDA_PATH=$CUDA_HOME
export CUDA_ROOT=$CUDA_HOME

export CPATH="$CUDA_HOME/include:$CUDA_HOME/extras/CUPTI/include:$CUDA_HOME/nvvm/include:$CPATH"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$LD_LIBRARY_PATH"

export PATH="$CUDA_HOME/bin:$CUDA_HOME/nvvm/bin:$CUDA_HOME:$PATH"

nvcc --version

