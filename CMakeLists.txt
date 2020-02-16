CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)

PROJECT(CudaVoxelize LANGUAGES CXX CUDA)

FIND_PACKAGE(GLM REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem program_options)
 
SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(Trimesh2_INCLUDE_DIR CACHE PATH "Path to Trimesh2 includes")
SET(CUDA_ARCH CACHE STRING "CUDA compute capability. It is prefer to set native value for your video card. Example: 61")

IF(NOT CUDA_ARCH)
	MESSAGE(FATAL_ERROR "You must set CUDA_ARCH variable. For example: 61")
ENDIF()

IF(NOT EXISTS "${Trimesh2_INCLUDE_DIR}")
  MESSAGE(FATAL_ERROR "You need set path to Trimesh2 include directory")
ENDIF()

FIND_FILE(Trimesh2_TriMesh_h TriMesh.h ${Trimesh2_INCLUDE_DIR})
IF(NOT Trimesh2_TriMesh_h)
	MESSAGE(FATAL_ERROR "Can't find TriMesh.h in ${Trimesh2_INCLUDE_DIR}")
ENDIF()

SET(Trimesh2_LINK_DIR CACHE PATH "Path to Trimesh2 libraries")

IF(NOT EXISTS "${Trimesh2_LINK_DIR}")
  MESSAGE(FATAL_ERROR "You need set path to Trimesh2 libraries")
ENDIF()

FIND_LIBRARY(Trimesh2_LIBRARY trimesh ${Trimesh2_LINK_DIR})

IF(NOT Trimesh2_LIBRARY)
	MESSAGE(FATAL_ERROR "Can't find librray in ${Trimesh2_LINK_DIR}")
ENDIF()

SET(CUDA_VOXELIZER_SRCS
  ./src/libs/helper_cuda.h
  ./src/libs/helper_string.h
  ./src/cpu_voxelizer.h
  ./src/cpu_voxelizer.cpp
  ./src/morton_LUTs.h
  ./src/thrust_operations.cuh
  ./src/thrust_operations.cu
  ./src/timer.h
  ./src/util_cuda.h
  ./src/util_cuda.cpp
  ./src/util_io.h
  ./src/util_io.cpp
  ./src/util.h
  ./src/util.cpp
  ./src/main.cpp
  ./src/voxelize.cuh
  ./src/voxelize.cu
  ./src/prepare_model.h
  ./src/prepare_model.cpp
  ./src/cpu_computing.h
  ./src/cpu_computing.cpp
  ./src/gpu_computing.h
  ./src/gpu_computing.cpp
)

ADD_EXECUTABLE(${CUDA_VOXELIZER_EXECUTABLE} ${CUDA_VOXELIZER_SRCS})
TARGET_COMPILE_FEATURES(${CUDA_VOXELIZER_EXECUTABLE} PUBLIC cxx_std_14)
TARGET_INCLUDE_DIRECTORIES(	${CUDA_VOXELIZER_EXECUTABLE} PUBLIC ${Trimesh2_INCLUDE_DIR})
TARGET_LINK_LIBRARIES ( ${CUDA_VOXELIZER_EXECUTABLE} PUBLIC ${Trimesh2_LIBRARY} PRIVATE glm PUBLIC Boost::filesystem PUBLIC Boost::program_options)
TARGET_COMPILE_OPTIONS(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}>)
