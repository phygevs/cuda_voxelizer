#include "prepare_model.h"

float* meshToGPU_managed(const trimesh::TriMesh* mesh)
{
	Timer t; t.start();
	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
	float* device_triangles;
	fprintf(stdout, "[Mesh] Allocating %s of CUDA-managed UNIFIED memory for triangle data \n", (readableSize(n_floats)).c_str());
	checkCudaErrors(cudaMallocManaged((void**)&device_triangles, n_floats)); // managed memory
	fprintf(stdout, "[Mesh] Copy %llu triangles to CUDA-managed UNIFIED memory \n", (size_t)(mesh->faces.size()));
	for (size_t i = 0; i < mesh->faces.size(); i++) {
		glm::vec3 v0 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][0]]);
		glm::vec3 v1 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][1]]);
		glm::vec3 v2 = trimesh_to_glm<trimesh::point>(mesh->vertices[mesh->faces[i][2]]);
		size_t j = i * 9;
		memcpy((device_triangles)+j, glm::value_ptr(v0), sizeof(glm::vec3));
		memcpy((device_triangles)+j + 3, glm::value_ptr(v1), sizeof(glm::vec3));
		memcpy((device_triangles)+j + 6, glm::value_ptr(v2), sizeof(glm::vec3));
	}
	t.stop();
	fprintf(stdout, "[Perf] Mesh transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);

	//for (size_t i = 0; i < mesh->faces.size(); i++) {
	//	size_t t = i * 9;
	//	std::cout << "Tri: " << device_triangles[t] << " " << device_triangles[t + 1] << " " << device_triangles[t + 2] << std::endl;
	//	std::cout << "Tri: " << device_triangles[t + 3] << " " << device_triangles[t + 4] << " " << device_triangles[t + 5] << std::endl;
	//	std::cout << "Tri: " << device_triangles[t + 6] << " " << device_triangles[t + 7] << " " << device_triangles[t + 8] << std::endl;
	//}

	return device_triangles;
}

bool voxelize_model(const boost::filesystem::path& path_to_model, const glm::uvec3& grid_sizes, unsigned int* vtable, std::size_t vtable_size, bool use_trust_path, OutputFormat output_format, bool force_cpu)
{
	assert(grid_sizes.x == grid_sizes.y == grid_sizes.z);
	Timer t;

	t.start();
	// SECTION: Read the mesh from disk using the TriMesh library
	fprintf(stdout, "\n## READ MESH \n");
#ifdef _DEBUG
	trimesh::TriMesh::set_verbose(true);
#endif
	cout << "[I/O] Reading mesh from " << path_to_model;

	unique_ptr<trimesh::TriMesh> themesh{ trimesh::TriMesh::read(path_to_model.string()) };

	if (themesh.get() == nullptr)
	{
		cerr << "Cannot load mesh" << endl;
		return false;
	}

	themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
	fprintf(stdout, "[Mesh] Number of triangles: %zu \n", themesh->faces.size());
	fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
	fprintf(stdout, "[Mesh] Computing bbox \n");
	themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

	// SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox
	AABox<glm::vec3> bbox_mesh(trimesh_to_glm(themesh->bbox.min), trimesh_to_glm(themesh->bbox.max));
	// Transform that AABox to a cubical box (by padding directions if needed)
	// Create voxinfo struct, which handles all the rest
	voxinfo voxelization_info(createMeshBBCube<glm::vec3>(bbox_mesh), grid_sizes, themesh->faces.size());
	voxelization_info.print();

	// SECTION: The actual voxelization
	if (!force_cpu)
	{
		// GPU voxelization
		fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");
		float* device_triangles{};
		// Transfer triangles to GPU using either thrust or managed cuda memory
		if (use_trust_path)
		{
			device_triangles = meshToGPU_thrust(themesh.get());
		}
		else
		{
			device_triangles = meshToGPU_managed(themesh.get());
		}

		fprintf(stdout, "\n## GPU VOXELISATION \n");
		voxelize(voxelization_info, device_triangles, vtable, use_trust_path, (output_format == OutputFormat::output_morton));

		if (use_trust_path)
		{
			cleanup_thrust();
		}
	}
	else
	{
		// CPU VOXELIZATION FALLBACK
		fprintf(stdout, "\n## CPU VOXELISATION \n");
		cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh.get(), vtable, (output_format == OutputFormat::output_morton));
	}

	fprintf(stdout, "\n## FILE OUTPUT \n");

	string output_name = (path_to_model.parent_path() / path_to_model.stem()).string();

	switch (output_format)
	{
	case output_morton:
		write_binary(vtable, vtable_size, output_name);
		break;
	case output_binvox:
		write_binvox(vtable, grid_sizes.x, output_name);
		break;
	case output_obj:
		write_obj(vtable, grid_sizes.x, output_name);
		break;
	default:
		cerr << "Unknown format. Cannot save result" << endl;
		return false;
	}

	fprintf(stdout, "\n## STATS \n");
	t.stop();
	fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);

	return true;
}