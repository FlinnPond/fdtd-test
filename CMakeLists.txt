cmake_minimum_required(VERSION 3.12)
project(fdtd LANGUAGES CXX CUDA)

find_package(
    CUDA
)

add_executable(fdtd 
    main.cu
    fdtd.cu
    params.cu
)

set_target_properties(fdtd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
