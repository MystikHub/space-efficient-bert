cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SpaceEfficientTransformer LANGUAGES CXX CUDA)

set(SEBERT_CMAKE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SEBERT_CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

find_package(spdlog REQUIRED)
find_package(CUDA 11.8 REQUIRED)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

include_directories(include /opt/cuda/include/)
link_directories(/opt/cuda/targets/x86_64-linux/lib/)

file(GLOB_RECURSE SRC_CPPS "${PROJECT_SOURCE_DIR}/src/*.cpp")
set(SEBERT_CMAKE_SOURCES
  ${SRC_CPPS}
  main.cpp
)

set(CMAKE_CXX_STANDARD 23)

add_executable(${PROJECT_NAME} ${SEBERT_CMAKE_SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog cuda cudart cudnn cublas)