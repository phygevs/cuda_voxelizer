#pragma once

#include <string>
#include <iosfwd>
#include <fstream>
#include <array>
#include "loggers.h"

const std::array<std::string, 4> valid_extensions{ ".ply", ".obj", ".3ds", ".off" };

size_t get_file_length(const std::string base_filename);
void read_binary(void* data, const size_t length, const std::string base_filename);
void write_binary(const void* data, const size_t bytes, const std::string base_filename);
void write_binvox(const unsigned int* vtable, const size_t gridsize, const std::string base_filename);
void write_obj(const unsigned int* vtable, const size_t gridsize, const std::string base_filename);

void save_voxel(const std::string& path, const unsigned int* vtable, const size_t vtable_size, const size_t grid_size, outfmt::OutputFormat output_format);
