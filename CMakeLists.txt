CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)

PROJECT(CudaVoxelize LANGUAGES CXX CUDA)

SET(CXX_STANDARD 14)

FIND_PACKAGE(GLM REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem program_options)
 
SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(Trimesh2_INCLUDE_DIR CACHE PATH "Path to Trimesh2 includes")

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
  ./src/main.cpp
  ./src/voxelize.cuh
  ./src/voxelize.cu
)

ADD_EXECUTABLE(${CUDA_VOXELIZER_EXECUTABLE} ${CUDA_VOXELIZER_SRCS})
TARGET_INCLUDE_DIRECTORIES(	${CUDA_VOXELIZER_EXECUTABLE} PUBLIC ${Trimesh2_INCLUDE_DIR})
TARGET_LINK_LIBRARIES ( ${CUDA_VOXELIZER_EXECUTABLE} PUBLIC ${Trimesh2_LIBRARY} PRIVATE glm PUBLIC Boost::filesystem PUBLIC Boost::program_options)
