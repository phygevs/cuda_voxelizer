CMAKE_MINIMUM_REQUIRED(VERSION 3.10 FATAL_ERROR)

PROJECT(CudaVoxelize LANGUAGES CXX CUDA)

FIND_PACKAGE(glm REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem program_options log log_setup)
 
SET(CUDA_VOXELIZER_EXECUTABLE cuda_voxelizer)

SET(Trimesh2_INCLUDE_DIR CACHE PATH "Path to Trimesh2 includes")
SET(CUDA_ARCH CACHE STRING "CUDA compute capability. It is prefer to set native value for your video card. Example: 61")

IF(NOT CUDA_ARCH)
	MESSAGE(FATAL_ERROR "You must set CUDA_ARCH variable. For example: 61")
ENDIF()

IF(NOT Trimesh2_INCLUDE_DIR)
  MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_INCLUDE_DIR")
ENDIF()

IF(NOT EXISTS "${Trimesh2_INCLUDE_DIR}")
  MESSAGE(FATAL_ERROR "You need set path to Trimesh2 include directory")
ENDIF()

FIND_FILE(Trimesh2_TriMesh_h TriMesh.h ${Trimesh2_INCLUDE_DIR})
IF(NOT Trimesh2_TriMesh_h)
	MESSAGE(FATAL_ERROR "Can't find TriMesh.h in ${Trimesh2_INCLUDE_DIR}")
ENDIF()

SET(Trimesh2_LINK_DIR CACHE PATH "Path to Trimesh2 libraries")

IF(NOT Trimesh2_LINK_DIR)
  MESSAGE(FATAL_ERROR "You need to set variable Trimesh2_LINK_DIR")
ENDIF()

IF(NOT EXISTS "${Trimesh2_LINK_DIR}")
  MESSAGE(FATAL_ERROR "You need to set path to Trimesh2 libraries")
ENDIF()

FIND_LIBRARY(Trimesh2_LIBRARY trimesh ${Trimesh2_LINK_DIR})

IF(NOT Trimesh2_LIBRARY)
	MESSAGE(FATAL_ERROR "Can't find librray in ${Trimesh2_LINK_DIR}")
ENDIF()

MESSAGE(STATUS "CUDA compute capability set to ${CUDA_ARCH}")
MESSAGE(STATUS "Found Trimesh2 include: ${Trimesh2_TriMesh_h}")
MESSAGE(STATUS "Found Trimesh2 lib: ${Trimesh2_LIBRARY}")


SET(CUDA_VOXELIZER_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/helper_cuda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/helper_string.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_voxelizer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_voxelizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/morton_LUTs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/thrust_operations.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/src/thrust_operations.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/timer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util_cuda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util_cuda.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util_io.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util_io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/voxelize.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/src/voxelize.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/prepare_model.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/prepare_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_computing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_computing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_computing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_computing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/loggers.h
)

ADD_EXECUTABLE(${CUDA_VOXELIZER_EXECUTABLE} ${CUDA_VOXELIZER_SRCS})
TARGET_COMPILE_FEATURES(${CUDA_VOXELIZER_EXECUTABLE} PUBLIC cxx_std_14)
TARGET_INCLUDE_DIRECTORIES(	${CUDA_VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_INCLUDE_DIR})
TARGET_LINK_LIBRARIES ( ${CUDA_VOXELIZER_EXECUTABLE} PRIVATE ${Trimesh2_LIBRARY} PRIVATE glm PRIVATE Boost::filesystem PRIVATE Boost::program_options PRIVATE Boost::log_setup PRIVATE Boost::log)
TARGET_COMPILE_OPTIONS(${CUDA_VOXELIZER_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}>)
