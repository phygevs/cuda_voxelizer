#include "prepare_model.h"

std::unique_ptr<trimesh::TriMesh> util::prepare_model(const boost::filesystem::path & path_to_model)
{
    logging::logger_t & logger_main = logging::logger_main::get();

    // SECTION: Read the mesh from disk using the TriMesh library
    BOOST_LOG_SEV(logger_main, logging::severity_t::info) << "## READ MESH" << endl;
#ifdef _DEBUG
    trimesh::TriMesh::set_verbose(true);
#endif
    BOOST_LOG_SEV(logger_main, logging::severity_t::debug) << "[I/O] Reading mesh from " << path_to_model << endl;

    trimesh::TriMesh * themesh = trimesh::TriMesh::read(path_to_model.string());

    if (themesh == nullptr)
    {
        BOOST_LOG_SEV(logger_main, logging::severity_t::error) << "Cannot load mesh" << endl;
        return unique_ptr<trimesh::TriMesh>(themesh);
    }

    themesh->need_faces(); // Trimesh: Unpack (possible) triangle strips so we have faces for sure
    BOOST_LOG_SEV(logger_main, logging::severity_t::debug) << "[Mesh] Number of triangles: " << themesh->faces.size() << endl;
    BOOST_LOG_SEV(logger_main, logging::severity_t::debug) << "[Mesh] Number of vertices: " << themesh->vertices.size() << endl;
    BOOST_LOG_SEV(logger_main, logging::severity_t::debug) << "[Mesh] Computing bbox" << endl;

    themesh->need_bbox(); // Trimesh: Compute the bounding box (in model coordinates)

    return unique_ptr<trimesh::TriMesh>(themesh);
}

voxinfo util::voxelization_setup(const trimesh::TriMesh * mesh, const glm::uvec3 & grid_sizes)
{
    logging::logger_t & logger_main = logging::logger_main::get();

    // SECTION: Compute some information needed for voxelization (bounding box, unit vector, ...)
    BOOST_LOG_SEV(logger_main, logging::severity_t::info) << "## VOXELISATION SETUP" << endl;
    // Initialize our own AABox
    AABox<glm::vec3> bbox_mesh(trimesh_to_glm(mesh->bbox.min), trimesh_to_glm(mesh->bbox.max));
    // Transform that AABox to a cubical box (by padding directions if needed)
    // Create voxinfo struct, which handles all the rest
    return voxinfo(createMeshBBCube<glm::vec3>(bbox_mesh), grid_sizes, mesh->faces.size());
}