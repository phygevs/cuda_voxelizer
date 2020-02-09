#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN // Please, not too much windows shenanigans
#endif

// Standard libs
#include <string>
#include <cstdio>
#include <sstream>
#include <map>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "prepare_model.h"

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;

constexpr char* version_number = "v0.4.4";

const map<string, OutputFormat> formats{
	{ "binvox", OutputFormat::output_binvox},
	{ "morton", OutputFormat::output_morton},
	{ "obj",  OutputFormat::output_obj}
};

constexpr char* input_arg_name{ "input" };
constexpr char* input_arg_short_name{ "i" };
constexpr char* voxel_resol_arg_name{ "vres" };
constexpr char* voxel_resol_arg_short_name{ "s" };
constexpr char* output_fromat_arg_name{ "oformat" };
constexpr char* output_fromat_arg_short_name{ "o" };
constexpr char* cpu_arg_name{ "cpu" };
constexpr char* trust_lib_arg_name{ "cutrust" };
constexpr char* trust_lib_arg_short_name{ "t" };

istream& operator>>(istream& in, OutputFormat& format)
{
	string token;

	in >> token;

	try
	{
		format = formats.at(token);
	}
	catch (out_of_range & exc)
	{
		in.setstate(ios_base::failbit);
	}

	return in;
}

auto parse_cli_args(int argc, char** argv)
{
	string model{ input_arg_name };
	model += ',';
	model += input_arg_short_name;

	string voxel_resol{ voxel_resol_arg_name };
	voxel_resol += ',';
	voxel_resol += voxel_resol_arg_short_name;

	string format_arg{ output_fromat_arg_name };
	format_arg += ',';
	format_arg += output_fromat_arg_short_name;

	string cuda_trust_arg{ trust_lib_arg_name };
	cuda_trust_arg += ',';
	cuda_trust_arg += trust_lib_arg_short_name;

	ostringstream description;
	description << "## CUDA VOXELIZER\n" << "CUDA Voxelizer" << version_number << " by Jeroen Baert" << endl << "Original code: https://github.com/Forceflow/cuda_voxelizer - mail@jeroen-baert.be" << endl << "Fork: https://github.com/KernelA/cuda_voxelizer" << endl << "Example: cuda_voxelizer -f bunny.ply -s 512 -t" << endl << "Options";

	options_description desc{ description.str() };

	ostringstream all_formats;

	all_formats << "output format: ";

	for (const auto& name : formats)
	{
		all_formats << name.first << ' ';
	}

	desc.add_options()
		("help,h", "Print help")
		(model.c_str(), value<string>(), "<path to model file: .ply, .obj, .3ds> (required). If it is directory then recursive voxelize all models in current directory in all subdirectories.")
		(voxel_resol.c_str(), value<int>()->default_value(128), "<voxelization grid size, power of 2: 8 -> 512, 1024, ... >")
		(format_arg.c_str(), value<OutputFormat>()->default_value(OutputFormat::output_binvox), all_formats.str().c_str())
		(cuda_trust_arg.c_str(), bool_switch(), "Force using CUDA Thrust Library (possible speedup / throughput improvement)")
		(cpu_arg_name, bool_switch(), "Force voxelization on the CPU instead of GPU.For when a CUDA device is not detected / compatible, or for very small models where GPU call overhead is not worth it")
		;

	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);

	return make_tuple(vm, desc);
}

bool validate_args(const variables_map& args, const options_description& desc)
{
	if (args.count(input_arg_name) != 1)
	{
		cerr << input_arg_name << " required argument" << endl;
		return false;
	}

	if (args.count(input_arg_name))
	{
		path input_file{ args[input_arg_name].as<string>() };

		auto input_type{ status(input_file).type() };

		if (input_type != file_type::regular_file && input_type != file_type::directory_file)
		{
			cerr << input_file << " is not file or directory. It may be no exist." << endl;
			return false;
		}
	}

	int voxel_resol{ args[voxel_resol_arg_name].as<int>() };

	if (voxel_resol <= 0)
	{
		cerr << "Voxel grid size order must be positive. Actual value is " << voxel_resol << endl;
		return false;
	}

	return true;
}

int main(int argc, char* argv[])
{
	variables_map args;

	try
	{
		auto res_tuple = parse_cli_args(argc, argv);
		args = get<0>(res_tuple);
		auto desc = get<1>(res_tuple);

		if (args.count("help"))
		{
			cout << desc << endl;
			return 0;
		}

		if (!validate_args(args, desc))
		{
			cout << desc << endl;
			return 1;
		}
	}
	catch (error & err)
	{
		cerr << err.what() << endl;
		return 1;
	}

	path input{ args[input_arg_name].as<string>() };
	bool forceCPU{ args[cpu_arg_name].as<bool>() };
	bool useThrustPath{ args[trust_lib_arg_name].as<bool>() };
	int gridsize{ args[voxel_resol_arg_name].as<int>() };
	OutputFormat outputformat{ args[output_fromat_arg_name].as< OutputFormat >() };

	// SECTION: Try to figure out if we have a CUDA-enabled GPU
	fprintf(stdout, "\n## CUDA INIT \n");
	bool cuda_ok = initCuda();

	if (cuda_ok) {
		fprintf(stdout, "[Info] CUDA GPU found\n");
	}
	else {
		fprintf(stdout, "[Info] CUDA GPU not found\n");
	}

	glm::uvec3 grid_sizes{ gridsize, gridsize, gridsize };

	// Compute space needed to hold voxel table (1 voxel / bit)
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(grid_sizes.x)* static_cast<size_t>(grid_sizes.y)* static_cast<size_t>(grid_sizes.z)) / 8.0f);

	unsigned int* vtable{}; // Both voxelization paths (GPU and CPU) need this

	if (cuda_ok && !forceCPU)
	{
		string readable_size{ readableSize(vtable_size) };

		if (!useThrustPath) {
			fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readable_size.c_str());
			checkCudaErrors(cudaMallocManaged((void**)&vtable, vtable_size));
		}
		else {
			// ALLOCATE MEMORY ON HOST
			fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n", readable_size.c_str());
			checkCudaErrors(cudaHostAlloc((void**)&vtable, vtable_size, cudaHostAllocDefault));
		}
	}
	else
	{
		if (!forceCPU)
		{
			fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n");
			forceCPU = true;
		}
		else
		{
			fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n");
		}

		vtable = (unsigned int*)calloc(1, vtable_size);
	}

	if (vtable == nullptr)
	{
		cerr << "[Error] Cannot allocate memory" << endl;
		return 1;
	}

	auto input_type{ status(input).type() };

	if (input_type == file_type::regular_file)
	{
		cout << "Voxelize model" << endl;

		if (!voxelize_model(input, grid_sizes, vtable, vtable_size, useThrustPath, outputformat, forceCPU))
		{
			cerr << "Cannot voxelize model" << endl;
		}
	}
	else if (input_type == file_type::directory_file)
	{
		cout << "Recursive voxelize all models" << endl;

		auto iterator = recursive_directory_iterator(input);

		const array<string, 4> valid_extensions{ ".ply", ".obj", ".3ds", ".off" };

		for (const auto& entry : iterator)
		{
			if (entry.status().type() == file_type::regular_file)
			{
				path local_file{ entry.path() };

				auto val = find(valid_extensions.cbegin(), valid_extensions.cend(), local_file.extension());

				if (val != valid_extensions.cend())
				{
					cout << "\tFound " << local_file << endl;

					if (!voxelize_model(local_file, grid_sizes, vtable, vtable_size, useThrustPath, outputformat, forceCPU))
					{
						cerr << "Cannot voxelize model" << endl;
					}

					if (forceCPU)
					{
						std::memset((void*)vtable, 0, vtable_size * sizeof(unsigned int));
					}
					else
					{
						checkCudaErrors(cudaMemset((void*)vtable, 0, vtable_size));
					}
				}
			}
		}
	}

	if (forceCPU)
	{
		free(vtable);
	}
	else
	{
		if (!useThrustPath)
		{
			cudaFree(vtable);
		}
		else
		{
			cudaFreeHost(vtable);
		}
	}

	return 0;
}