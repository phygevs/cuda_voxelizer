#include "gpu_computing.h"

void gpu::compute_voxels(const boost::filesystem::path& input, const glm::uvec3& grid_sizes, std::size_t vtable_size, outfmt::OutputFormat output_format, bool use_trust)
{
	using namespace boost::filesystem;

	assert(vtable_size > 0);

	string readable_size{ readableSize(vtable_size) };

	auto cuda_deleter = [use_trust](unsigned int* ptr) -> void
	{
		if (use_trust)
		{
			checkCudaErrors(cudaFreeHost(ptr));
		}
		else
		{
			checkCudaErrors(cudaFree(ptr));
		}
	};

	auto cuda_malloc = [use_trust, &readable_size](size_t size) -> unsigned int*
	{
		unsigned int* ptr{};

		cudaError_t error{ cudaSuccess };

		if (use_trust)
		{
			// ALLOCATE MEMORY ON HOST
			BOOST_LOG_TRIVIAL(debug) << "[Voxel Grid] Allocating " << readable_size << " kB of page-locked HOST memory for Voxel Grid" << endl;

			error = cudaHostAlloc((void**)&ptr, size, cudaHostAllocDefault);
		}
		else
		{
			BOOST_LOG_TRIVIAL(debug) << "[Voxel Grid] Allocating " << readable_size << " of CUDA-managed UNIFIED memory for Voxel Grid" << endl;

			error = cudaMallocManaged((void**)&ptr, size);
		}

		if (error != cudaSuccess)
		{
			BOOST_LOG_TRIVIAL(error) << cudaGetErrorString(error) << endl;
		}

		return ptr;
	};

	std::unique_ptr<unsigned int, decltype(cuda_deleter)> vtable{ cuda_malloc(vtable_size), cuda_deleter };

	if (vtable.get() == nullptr)
	{
		BOOST_LOG_TRIVIAL(error) << "Cannot allocate memory on GPU" << endl;
		return;
	}

	auto input_type{ status(input).type() };

	if (input_type == file_type::regular_file)
	{
		BOOST_LOG_TRIVIAL(info) << "Voxelize model" << endl;

		if (!prepare_model_and_voxelize(input, vtable.get(), vtable_size, grid_sizes, output_format, use_trust))
		{
			BOOST_LOG_TRIVIAL(error) << "Cannot voxelize model " << input << endl;
		}
	}
	else if (input_type == file_type::directory_file)
	{
		BOOST_LOG_TRIVIAL(info) << "Recursive voxelize all models" << endl;

		auto iterator = recursive_directory_iterator(input);

		for (const auto& entry : iterator)
		{
			if (entry.status().type() == file_type::regular_file)
			{
				path local_file{ entry.path() };

				auto val = find(valid_extensions.cbegin(), valid_extensions.cend(), local_file.extension());

				if (val != valid_extensions.cend())
				{
					BOOST_LOG_TRIVIAL(info) << "Found " << local_file << endl;

					if (!prepare_model_and_voxelize(local_file, vtable.get(), vtable_size, grid_sizes, output_format, use_trust))
					{
						BOOST_LOG_TRIVIAL(error) << "Cannot voxelize model" << endl;
					}

					checkCudaErrors(cudaMemset((void*)vtable.get(), 0, vtable_size));
				}
			}
		}
	}
}

bool gpu::prepare_model_and_voxelize(const boost::filesystem::path& path_to_model, unsigned int* vtable, std::size_t vtable_size, const glm::uvec3& grid_sizes, const outfmt::OutputFormat output_format, bool use_trust)
{
	assert(grid_sizes.x == grid_sizes.y == grid_sizes.z);

	Timer t;

	t.start();

	unique_ptr<trimesh::TriMesh> themesh{ util::prepare_model(path_to_model) };

	if (themesh.get() == nullptr)
	{
		BOOST_LOG_TRIVIAL(error) << "Cannot load mesh" << endl;
		return false;
	}

	voxinfo voxelization_info{ util::voxelization_setup(themesh.get(), grid_sizes) };

	// GPU voxelization
	BOOST_LOG_TRIVIAL(debug) << "## TRIANGLES TO GPU TRANSFER" << endl;

	float* device_triangles{};

	// Transfer triangles to GPU using either thrust or managed cuda memory
	if (use_trust)
	{
		device_triangles = meshToGPU_thrust(themesh.get());
	}
	else
	{
		device_triangles = meshToGPU_managed(themesh.get());
	}

	BOOST_LOG_TRIVIAL(debug) << "## GPU VOXELISATION" << endl;

	compute_voxels(voxelization_info, device_triangles, vtable, use_trust, (output_format == outfmt::OutputFormat::output_morton));

	if (use_trust)
	{
		cleanup_thrust();
	}
	else
	{
		checkCudaErrors(cudaFree(device_triangles));
	}

	BOOST_LOG_TRIVIAL(debug) << "## FILE OUTPUT" << endl;

	string output_name = (path_to_model.parent_path() / path_to_model.stem()).string();

	save_voxel(output_name, vtable, vtable_size, grid_sizes.x, output_format);

	t.stop();
	BOOST_LOG_TRIVIAL(debug) << "## STATS" << endl;
	BOOST_LOG_TRIVIAL(debug) << "[Perf] Total runtime: " << t.elapsed_time_milliseconds << " ms" << endl;

	return true;
}

float* gpu::meshToGPU_managed(const trimesh::TriMesh* mesh)
{
	Timer t;
	t.start();
	size_t n_floats = sizeof(float) * 9 * (mesh->faces.size());
	float* device_triangles{};

	BOOST_LOG_TRIVIAL(debug) << "[Mesh] Allocating " << readableSize(n_floats) << " of CUDA-managed UNIFIED memory for triangle data" << endl;
	checkCudaErrors(cudaMallocManaged((void**)&device_triangles, n_floats)); // managed memory
	BOOST_LOG_TRIVIAL(debug) << "[Mesh] Copy " << mesh->faces.size() << " triangles to CUDA-managed UNIFIED memory" << endl;
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
	BOOST_LOG_TRIVIAL(debug) << "[Perf] Mesh transfer time to GPU: " << t.elapsed_time_milliseconds << " ms" << endl;

	//for (size_t i = 0; i < mesh->faces.size(); i++) {
	//	size_t t = i * 9;
	//	std::cout << "Tri: " << device_triangles[t] << " " << device_triangles[t + 1] << " " << device_triangles[t + 2] << std::endl;
	//	std::cout << "Tri: " << device_triangles[t + 3] << " " << device_triangles[t + 4] << " " << device_triangles[t + 5] << std::endl;
	//	std::cout << "Tri: " << device_triangles[t + 6] << " " << device_triangles[t + 7] << " " << device_triangles[t + 8] << std::endl;
	//}

	return device_triangles;
}