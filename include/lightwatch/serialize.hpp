// Phase 37: Serialization
// Model weight serialization in .lwbin format

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/tensor.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

namespace lightwatch {

// .lwbin file format constants
constexpr char LWBIN_MAGIC[4] = {'L', 'W', 'A', 'I'};
constexpr uint32_t LWBIN_VERSION = 1;
constexpr size_t LWBIN_HEADER_SIZE = 64;

// Data types
enum class DType : uint8_t {
    FLOAT32 = 0,
    FLOAT16 = 1,
    INT32 = 2
};

// Tensor metadata
struct TensorMetadata {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype;
    size_t data_offset;
    size_t data_size;
};

// Serialization error
struct SerializeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// File header structure
struct LwbinHeader {
    char magic[4];
    uint32_t version;
    uint32_t tensor_count;
    uint8_t reserved[52];
};

// Save weights to .lwbin file
inline void save_weights(const std::string& path, const nn::Module& model) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw SerializeError("Cannot open file for writing: " + path);
    }

    auto state_dict = model.state_dict();

    // Write header
    LwbinHeader header = {};
    std::memcpy(header.magic, LWBIN_MAGIC, 4);
    header.version = LWBIN_VERSION;
    header.tensor_count = static_cast<uint32_t>(state_dict.size());

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write tensor metadata (name length, name, ndims, shape, dtype, data size)
    // Then write tensor data
    for (const auto& [name, tensor] : state_dict) {
        // Name length and name
        uint32_t name_len = static_cast<uint32_t>(name.size());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.data(), static_cast<std::streamsize>(name_len));

        // Number of dimensions
        const auto& shape = tensor.shape();
        uint32_t ndims = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));

        // Shape
        for (size_t dim : shape) {
            int64_t dim64 = static_cast<int64_t>(dim);
            file.write(reinterpret_cast<const char*>(&dim64), sizeof(dim64));
        }

        // Data type (always float32 for now)
        uint8_t dtype = static_cast<uint8_t>(DType::FLOAT32);
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        // Data size
        uint64_t data_size = tensor.numel() * sizeof(float);
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

        // Tensor data
        file.write(reinterpret_cast<const char*>(tensor.data()), static_cast<std::streamsize>(data_size));
    }

    if (!file.good()) {
        throw SerializeError("Error writing to file: " + path);
    }
}

// Load weights from .lwbin file
inline void load_weights(const std::string& path, nn::Module& model) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw SerializeError("Cannot open file for reading: " + path);
    }

    // Read header
    LwbinHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // Validate magic
    if (std::memcmp(header.magic, LWBIN_MAGIC, 4) != 0) {
        throw SerializeError("Invalid file format: bad magic number");
    }

    // Validate version
    if (header.version > LWBIN_VERSION) {
        throw SerializeError("Unsupported file version: " + std::to_string(header.version));
    }

    // Get model's state dict for validation
    auto state_dict = model.state_dict();

    // Read tensors
    for (uint32_t i = 0; i < header.tensor_count; ++i) {
        // Name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data(), static_cast<std::streamsize>(name_len));

        // Number of dimensions
        uint32_t ndims;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));

        // Shape
        std::vector<size_t> shape(ndims);
        for (uint32_t d = 0; d < ndims; ++d) {
            int64_t dim64;
            file.read(reinterpret_cast<char*>(&dim64), sizeof(dim64));
            shape[d] = static_cast<size_t>(dim64);
        }

        // Data type
        uint8_t dtype;
        file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));

        // Data size
        uint64_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));

        // Find tensor in model
        auto it = state_dict.find(name);
        if (it == state_dict.end()) {
            // Skip unknown tensor
            file.seekg(static_cast<std::streamoff>(data_size), std::ios::cur);
            continue;
        }

        auto& tensor = it->second;

        // Validate shape
        if (tensor.shape() != shape) {
            throw SerializeError("Shape mismatch for tensor '" + name + "'");
        }

        // Read data
        if (dtype == static_cast<uint8_t>(DType::FLOAT32)) {
            file.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(data_size));
        } else {
            throw SerializeError("Unsupported dtype for tensor '" + name + "'");
        }
    }

    if (!file.good() && !file.eof()) {
        throw SerializeError("Error reading from file: " + path);
    }
}

// Inspect weights file without loading
inline std::vector<TensorMetadata> inspect_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw SerializeError("Cannot open file for reading: " + path);
    }

    // Read header
    LwbinHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (std::memcmp(header.magic, LWBIN_MAGIC, 4) != 0) {
        throw SerializeError("Invalid file format: bad magic number");
    }

    std::vector<TensorMetadata> result;
    result.reserve(header.tensor_count);

    size_t current_offset = sizeof(header);

    for (uint32_t i = 0; i < header.tensor_count; ++i) {
        TensorMetadata meta;

        // Name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        meta.name.resize(name_len);
        file.read(meta.name.data(), static_cast<std::streamsize>(name_len));
        current_offset += sizeof(name_len) + name_len;

        // Number of dimensions
        uint32_t ndims;
        file.read(reinterpret_cast<char*>(&ndims), sizeof(ndims));
        current_offset += sizeof(ndims);

        // Shape
        meta.shape.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d) {
            int64_t dim64;
            file.read(reinterpret_cast<char*>(&dim64), sizeof(dim64));
            meta.shape[d] = dim64;
        }
        current_offset += ndims * sizeof(int64_t);

        // Data type
        uint8_t dtype;
        file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
        meta.dtype = static_cast<DType>(dtype);
        current_offset += sizeof(dtype);

        // Data size
        uint64_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        meta.data_size = static_cast<size_t>(data_size);
        current_offset += sizeof(data_size);

        meta.data_offset = current_offset;
        current_offset += meta.data_size;

        // Skip data
        file.seekg(static_cast<std::streamoff>(meta.data_size), std::ios::cur);

        result.push_back(std::move(meta));
    }

    return result;
}

// Validate weights file against model architecture
inline bool validate_weights(const std::string& path, const nn::Module& model) {
    try {
        auto file_tensors = inspect_weights(path);
        auto state_dict = model.state_dict();

        // Check each tensor in file exists in model with matching shape
        for (const auto& meta : file_tensors) {
            auto it = state_dict.find(meta.name);
            if (it == state_dict.end()) {
                return false;  // Tensor not found in model
            }

            const auto& model_shape = it->second.shape();
            if (model_shape.size() != meta.shape.size()) {
                return false;  // Different number of dimensions
            }

            for (size_t i = 0; i < model_shape.size(); ++i) {
                if (static_cast<int64_t>(model_shape[i]) != meta.shape[i]) {
                    return false;  // Shape mismatch
                }
            }
        }

        return true;
    } catch (const SerializeError&) {
        return false;
    }
}

// Read header from file
inline LwbinHeader read_header(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw SerializeError("Cannot open file for reading: " + path);
    }

    LwbinHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!file.good()) {
        throw SerializeError("Error reading header from file: " + path);
    }

    return header;
}

// Check if file is valid .lwbin format
inline bool is_valid_lwbin(const std::string& path) {
    try {
        auto header = read_header(path);
        return std::memcmp(header.magic, LWBIN_MAGIC, 4) == 0 &&
               header.version <= LWBIN_VERSION;
    } catch (...) {
        return false;
    }
}

}  // namespace lightwatch
