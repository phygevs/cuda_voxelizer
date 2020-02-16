#pragma once

#include <array>
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
#include "timer.h"
// CPU voxelizer fallback
#include "cpu_voxelizer.h"
#include "prepare_model.h"

namespace cpu
{
	void compute_voxels(const boost::filesystem::path& input, const glm::vec3& grid_sizes, std::size_t vtable_size, outfmt::OutputFormat output_format);

	bool prepare_model_and_voxelize(const boost::filesystem::path& path_to_model, unsigned int* vtable, std::size_t vtable_size, const glm::vec3& grid_sizes, const outfmt::OutputFormat output_format);
}
