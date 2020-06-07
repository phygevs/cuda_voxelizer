#pragma once

#include <boost/filesystem.hpp>
#include <array>
#include "loggers.h"

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
#include "prepare_model.h"

// Forward declaration of CUDA functions
float * meshToGPU_thrust(const trimesh::TriMesh * mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void compute_voxels(const voxinfo & v, float * triangle_data, unsigned int * vtable, bool useThrustPath, bool morton_code);

namespace gpu
{
    // METHOD 1: Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
    float * meshToGPU_managed(const trimesh::TriMesh * mesh);

    void compute_voxels(const boost::filesystem::path & input, const glm::uvec3 & grid_sizes, std::size_t vtable_size, outfmt::OutputFormat output_format, bool use_trust);

    bool prepare_model_and_voxelize(const boost::filesystem::path & path_to_model, unsigned int * vtable, std::size_t vtable_size, const glm::uvec3 & grid_sizes, const outfmt::OutputFormat output_format, bool use_trust);
}