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
#include "timer.h"

namespace util
{
	std::unique_ptr<trimesh::TriMesh> prepare_model(const boost::filesystem::path& path_to_model);

	voxinfo voxelization_setup(const trimesh::TriMesh* mesh, const glm::uvec3& grid_sizes);
}