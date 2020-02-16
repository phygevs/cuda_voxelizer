#include "cpu_computing.h"

void cpu::compute_voxels(const boost::filesystem::path& input, const glm::uvec3& grid_sizes, std::size_t vtable_size, outfmt::OutputFormat output_format)
{
	using namespace boost::filesystem;

	logging::logger_t& logger_main = logging::logger_main::get();

	assert(vtable_size > 0);

	unique_ptr<unsigned int, decltype(&std::free)> vtable((unsigned int*)calloc(1, vtable_size), std::free);

	if (vtable.get() == nullptr)
	{
		BOOST_LOG_SEV(logger_main, logging::severity_t::error) << "Cannot allocate memory" << endl;
		return;
	}

	auto input_type{ status(input).type() };

	if (input_type == file_type::regular_file)
	{
		BOOST_LOG_SEV(logger_main, logging::severity_t::info) << "Voxelize model" << endl;

		if (!prepare_model_and_voxelize(input, vtable.get(), vtable_size, grid_sizes, output_format))
		{
			BOOST_LOG_SEV(logger_main, logging::severity_t::error) << "Cannot voxelize model " << input << endl;
		}
	}
	else if (input_type == file_type::directory_file)
	{
		BOOST_LOG_SEV(logger_main, logging::severity_t::info) << "Recursive voxelize all models" << endl;

		auto iterator = recursive_directory_iterator(input);

		for (const auto& entry : iterator)
		{
			if (entry.status().type() == file_type::regular_file)
			{
				path local_file{ entry.path() };

				auto val = find(valid_extensions.cbegin(), valid_extensions.cend(), local_file.extension());

				if (val != valid_extensions.cend())
				{
					BOOST_LOG_SEV(logger_main, logging::severity_t::info) << "Found " << local_file << endl;

					if (!prepare_model_and_voxelize(local_file, vtable.get(), vtable_size, grid_sizes, output_format))
					{
						BOOST_LOG_SEV(logger_main, logging::severity_t::error) << "Cannot voxelize model" << endl;
					}

					std::memset((void*)vtable.get(), 0, vtable_size);
				}
			}
		}
	}
}

bool cpu::prepare_model_and_voxelize(const boost::filesystem::path& path_to_model, unsigned int* vtable, std::size_t vtable_size, const glm::uvec3& grid_sizes, const outfmt::OutputFormat output_format)
{
	assert(grid_sizes.x == grid_sizes.y == grid_sizes.z);

	logging::logger_t& logger_main = logging::logger_main::get();

	Timer t;

	t.start();

	unique_ptr<trimesh::TriMesh> themesh{ util::prepare_model(path_to_model) };

	if (themesh.get() == nullptr)
	{
		BOOST_LOG_SEV(logger_main, logging::severity_t::error) << "Cannot load mesh" << endl;
		return false;
	}

	voxinfo voxelization_info{ util::voxelization_setup(themesh.get(), grid_sizes) };

	BOOST_LOG_SEV(logger_main, logging::severity_t::info) << "## CPU VOXELISATION" << endl;

	cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh.get(), vtable, (output_format == outfmt::OutputFormat::output_morton));

	BOOST_LOG_SEV(logger_main, logging::severity_t::info) << "## FILE OUTPUT" << endl;

	const string output_name{ (path_to_model.parent_path() / path_to_model.stem()).string() };

	save_voxel(output_name, vtable, vtable_size, grid_sizes.x, output_format);

	BOOST_LOG_SEV(logger_main, logging::severity_t::debug) << "## STATS" << endl;

	t.stop();
	BOOST_LOG_SEV(logger_main, logging::severity_t::debug) << "[Perf] Total runtime: " << t.elapsed_time_milliseconds << " ms" << endl;
	return true;
}