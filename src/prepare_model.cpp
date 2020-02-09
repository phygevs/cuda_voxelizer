#include "prepare_model.h"

std::unique_ptr<trimesh::TriMesh> util::prepare_model(const boost::filesystem::path& path_to_model)
{
	// SECTION: Read the mesh from disk using the TriMesh library
	fprintf(stdout, "\n## READ MESH \n");
#ifdef _DEBUG
	trimesh::TriMesh::set_verbose(true);
#endif
	cout << "[I/O] Reading mesh from " << path_to_model << endl;

	trimesh::TriMesh* themesh = trimesh::TriMesh::read(path_to_model.string());

	if (themesh == nullptr)
	{
		cerr << "Cannot load mesh" << endl;
		return unique_ptr<trimesh::TriMesh>(themesh);
	}

	themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
	fprintf(stdout, "[Mesh] Number of triangles: %zu \n", themesh->faces.size());
	fprintf(stdout, "[Mesh] Number of vertices: %zu \n", themesh->vertices.size());
	fprintf(stdout, "[Mesh] Computing bbox \n");
	themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

	return unique_ptr<trimesh::TriMesh>(themesh);
}

voxinfo util::voxelization_setup(const trimesh::TriMesh* mesh, const glm::vec3& grid_sizes)
{
	// SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
	fprintf(stdout, "\n## VOXELISATION SETUP \n");
	// Initialize our own AABox
	AABox<glm::vec3> bbox_mesh(trimesh_to_glm(mesh->bbox.min), trimesh_to_glm(mesh->bbox.max));
	// Transform that AABox to a cubical box (by padding directions if needed)
	// Create voxinfo struct, which handles all the rest
	return voxinfo(createMeshBBCube<glm::vec3>(bbox_mesh), grid_sizes, mesh->faces.size());
}