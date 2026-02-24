#!/bin/bash

source ~/anaconda3/bin/activate styleGANenv

export TORCH_CUDA_ARCH_LIST="8.0"  # assume to have an A100 GPU (compute capability 8.0)

export CUDA_HOME=/usr/local/cuda-12.5 # Set CUDA 12.5 environment variables
export PATH=/usr/local/cuda-12.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH





