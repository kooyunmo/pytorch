cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cuda_mem_prefetch_async)

find_package(Torch REQUIRED)

# Define our library target
add_library(cuda_mem_prefetch_async SHARED op.cpp)
# Enable C++14
target_compile_features(cuda_mem_prefetch_async PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(cuda_mem_prefetch_async "${TORCH_LIBRARIES}")