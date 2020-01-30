#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN // Please, not too much windows shenanigans
#endif

// Standard libs
#include <string>
#include <cstdio>
#include <sstream>
#include <map>

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

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;

constexpr char* version_number = "v0.4.4";

// Forward declaration of CUDA functions
float* meshToGPU_thrust(const trimesh::TriMesh* mesh); // METHOD 3 to transfer triangles can be found in thrust_operations.cu(h)
void cleanup_thrust();
void voxelize(const voxinfo& v, float* triangle_data, unsigned int* vtable, bool useThrustPath, bool morton_code);

// Output formats
enum  OutputFormat { output_binvox = 0, output_morton = 1, output_obj = 2 };

const map<string, OutputFormat> formats{
	{ "binvox", OutputFormat::output_binvox},
	{ "morton", OutputFormat::output_morton},
	{ "obj",  OutputFormat::output_obj}
};

constexpr char* input_file_arg_name{ "file" };
constexpr char* input_file_arg_short_name{ "f" };
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
	string model{ input_file_arg_name };
	model += ',';
	model += input_file_arg_short_name;

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
		(model.c_str(), value<string>(), "<path to model file: .ply, .obj, .3ds> (required)")
		(voxel_resol.c_str(), value<int>()->default_value(256), "<voxelization grid size, power of 2: 8 -> 512, 1024, ... >")
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
	if (args.count(input_file_arg_name) != 1)
	{
		cerr << input_file_arg_name << " required argument" << endl;
		return false;
	}

	path input_file{ args[input_file_arg_name].as<string>() };

	if (status(input_file).type() != file_type::regular_file)
	{
		cerr << input_file << " is not file or does not exist." << endl;
		return false;
	}

	int voxel_resol{ args[voxel_resol_arg_name].as<int>() };

	if (voxel_resol <= 0)
	{
		cerr << "Voxel grid size order must be positive. Actual value is " << voxel_resol << endl;
		return false;
	}

	return true;
}

// METHOD 1: Helper function to transfer triangles to automatically managed CUDA memory ( > CUDA 7.x)
float* meshToGPU_managed(const trimesh::TriMesh* mesh) {
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
	t.stop(); fprintf(stdout, "[Perf] Mesh transfer time to GPU: %.1f ms \n", t.elapsed_time_milliseconds);

	//for (size_t i = 0; i < mesh->faces.size(); i++) {
	//	size_t t = i * 9;
	//	std::cout << "Tri: " << device_triangles[t] << " " << device_triangles[t + 1] << " " << device_triangles[t + 2] << std::endl;
	//	std::cout << "Tri: " << device_triangles[t + 3] << " " << device_triangles[t + 4] << " " << device_triangles[t + 5] << std::endl;
	//	std::cout << "Tri: " << device_triangles[t + 6] << " " << device_triangles[t + 7] << " " << device_triangles[t + 8] << std::endl;
	//}

	return device_triangles;
}

int main(int argc, char* argv[]) {
	Timer t;
	t.start();

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

	trimesh::TriMesh::set_verbose(false);

	path input_file{ args[input_file_arg_name].as<string>() };
	bool forceCPU{ args[cpu_arg_name].as<bool>() }, useThrustPath{ args[trust_lib_arg_name].as<bool>() };
	int gridsize{ args[voxel_resol_arg_name].as<int>() };
	OutputFormat outputformat{ args[output_fromat_arg_name].as< OutputFormat >() };

	// SECTION: Read the mesh from disk using the TriMesh library
	fprintf(stdout, "\n## READ MESH \n");
#ifdef _DEBUG
	trimesh::TriMesh::set_verbose(true);
#endif
	cout << "[I/O] Reading mesh from %s \n" << input_file;

	trimesh::TriMesh* themesh = trimesh::TriMesh::read(input_file.string());

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
	voxinfo voxelization_info(createMeshBBCube<glm::vec3>(bbox_mesh), glm::uvec3(gridsize, gridsize, gridsize), themesh->faces.size());
	voxelization_info.print();
	// Compute space needed to hold voxel table (1 voxel / bit)
	size_t vtable_size = static_cast<size_t>(ceil(static_cast<size_t>(voxelization_info.gridsize.x)* static_cast<size_t>(voxelization_info.gridsize.y)* static_cast<size_t>(voxelization_info.gridsize.z)) / 8.0f);
	unsigned int* vtable; // Both voxelization paths (GPU and CPU) need this

	// SECTION: Try to figure out if we have a CUDA-enabled GPU
	fprintf(stdout, "\n## CUDA INIT \n");
	bool cuda_ok = initCuda();
	if (cuda_ok) {
		fprintf(stdout, "[Info] CUDA GPU found\n");
	}
	else {
		fprintf(stdout, "[Info] CUDA GPU not found\n");
	}

	// SECTION: The actual voxelization
	if (cuda_ok && !forceCPU) {
		// GPU voxelization
		fprintf(stdout, "\n## TRIANGLES TO GPU TRANSFER \n");

		float* device_triangles;
		// Transfer triangles to GPU using either thrust or managed cuda memory
		if (useThrustPath) { device_triangles = meshToGPU_thrust(themesh); }
		else { device_triangles = meshToGPU_managed(themesh); }

		if (!useThrustPath) {
			fprintf(stdout, "[Voxel Grid] Allocating %s of CUDA-managed UNIFIED memory for Voxel Grid\n", readableSize(vtable_size).c_str());
			checkCudaErrors(cudaMallocManaged((void**)&vtable, vtable_size));
		}
		else {
			// ALLOCATE MEMORY ON HOST
			fprintf(stdout, "[Voxel Grid] Allocating %s kB of page-locked HOST memory for Voxel Grid\n", readableSize(vtable_size).c_str());
			checkCudaErrors(cudaHostAlloc((void**)&vtable, vtable_size, cudaHostAllocDefault));
		}
		fprintf(stdout, "\n## GPU VOXELISATION \n");
		voxelize(voxelization_info, device_triangles, vtable, useThrustPath, (outputformat == OutputFormat::output_morton));
	}
	else {
		// CPU VOXELIZATION FALLBACK
		fprintf(stdout, "\n## CPU VOXELISATION \n");
		if (!forceCPU) { fprintf(stdout, "[Info] No suitable CUDA GPU was found: Falling back to CPU voxelization\n"); }
		else { fprintf(stdout, "[Info] Doing CPU voxelization (forced using command-line switch -cpu)\n"); }
		vtable = (unsigned int*)calloc(1, vtable_size);
		cpu_voxelizer::cpu_voxelize_mesh(voxelization_info, themesh, vtable, (outputformat == OutputFormat::output_morton));
	}

	//// DEBUG: print vtable
	//for (int i = 0; i < vtable_size; i++) {
	//	char* vtable_p = (char*)vtable;
	//	cout << (int) vtable_p[i] << endl;
	//}

	fprintf(stdout, "\n## FILE OUTPUT \n");
	if (outputformat == OutputFormat::output_morton) {
		write_binary(vtable, vtable_size, input_file.stem().string());
	}
	else if (outputformat == OutputFormat::output_binvox) {
		write_binvox(vtable, gridsize, input_file.stem().string());
	}
	else if (outputformat == OutputFormat::output_obj) {
		write_obj(vtable, gridsize, input_file.stem().string());
	}

	if (useThrustPath) {
		cleanup_thrust();
	}

	fprintf(stdout, "\n## STATS \n");
	t.stop();
	fprintf(stdout, "[Perf] Total runtime: %.1f ms \n", t.elapsed_time_milliseconds);
}