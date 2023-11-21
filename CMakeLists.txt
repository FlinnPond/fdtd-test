cmake_minimum_required(VERSION 3.12)
project(fdtd LANGUAGES CXX CUDA)

find_package(
    CUDA
)

add_executable(fdtd 
    main.cu
    fdtd.cu
    params.cu
    plot.cu
)
target_compile_features(fdtd PRIVATE cxx_std_17)

set_target_properties(fdtd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)
target_link_libraries(fdtd PRIVATE 
    boost_iostreams boost_system boost_filesystem
)