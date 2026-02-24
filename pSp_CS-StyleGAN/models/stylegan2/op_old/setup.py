from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Set correct CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.4'  # Updated for CUDA 12.4
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"
os.environ['CC'] = '/usr/bin/gcc-11'
os.environ['CXX'] = '/usr/bin/g++-11'

setup(
    name='stylegan2_ops',
    ext_modules=[
        CUDAExtension(
            name='fused',
            sources=['fused_bias_act.cpp', 'fused_bias_act_kernel.cu'],
            extra_compile_args={
                'cxx': [
                    '-std=c++11',
                    '-D_GLIBCXX_USE_CXX11_ABI=0'
                ],
                'nvcc': [
                    '-std=c++11',
                    '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '-Xcompiler',
                    '-fPIC',
                    '--ptxas-options=-v',
                    '--gpu-architecture=compute_80',
                    '--gpu-code=sm_80',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__'
                ]
            }
        ),
        CUDAExtension(
            name='upfirdn2d',
            sources=['upfirdn2d.cpp', 'upfirdn2d_kernel.cu'],
            extra_compile_args={
                'cxx': [
                    '-std=c++11',
                    '-D_GLIBCXX_USE_CXX11_ABI=0'
                ],
                'nvcc': [
                    '-std=c++11',
                    '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '-Xcompiler',
                    '-fPIC',
                    '--ptxas-options=-v',
                    '--gpu-architecture=compute_80',
                    '--gpu-code=sm_80',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)