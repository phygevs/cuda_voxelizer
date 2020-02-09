#pragma once

#include <boost/filesystem.hpp>
// GLM for maths
#define GLM_FORCE_PURE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
// Trimesh for model importing
#include "TriMesh.h"
// Util
#include "util.h"
#include "util_io.h"
#include "util_cuda.h"
#include "timer.h"
// CPU voxelizer fallback
#include "cpu_voxelizer.h"

// Forward declaration of CUDA functions
float* meshToGPU_thrust(const trimesh::TriMesh* mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void voxelize(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);

// Output formats
enum  OutputFormat { output_binvox = 0, output_morton = 1, output_obj = 2 };

// METHOD 1: Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed(const trimesh::TriMesh* mesh);

bool voxelize_model(const boost::filesystem::path& path_to_model, const glm::uvec3& grid_size, unsigned int* vtable, std::size_t vtable_size, bool use_trust_path, OutputFormat output_format, bool force_cpu);