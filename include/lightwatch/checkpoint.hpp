// Phase 26: Checkpointing

#pragma once

#include <lightwatch/tensor.hpp>
#include <lightwatch/nn/module.hpp>
#include <lightwatch/optim/optimizer.hpp>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <vector>

namespace lightwatch {

// Checkpoint data structure
struct Checkpoint {
    std::unordered_map<std::string, Tensor<float>> model_state;
    std::unordered_map<std::string, Tensor<float>> optimizer_state;
    int epoch = 0;
    int step = 0;
    float loss = 0.0f;
    std::string config_json;
};

namespace detail {

// Magic number for checkpoint files
constexpr uint32_t CKPT_MAGIC = 0x4C574B54;  // "LWKT"
constexpr uint32_t CKPT_VERSION = 1;

// Write a tensor to binary stream
inline void write_tensor(std::ofstream& out, const std::string& name, const Tensor<float>& tensor) {
    // Write name length and name
    uint32_t name_len = static_cast<uint32_t>(name.size());
    out.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
    out.write(name.data(), name_len);

    // Write number of dimensions
    uint32_t ndim = static_cast<uint32_t>(tensor.ndim());
    out.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

    // Write shape
    for (size_t i = 0; i < tensor.ndim(); ++i) {
        uint64_t dim = static_cast<uint64_t>(tensor.shape()[i]);
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }

    // Write data
    uint64_t numel = tensor.numel();
    out.write(reinterpret_cast<const char*>(&numel), sizeof(numel));
    out.write(reinterpret_cast<const char*>(tensor.data()), numel * sizeof(float));
}

// Read a tensor from binary stream
inline bool read_tensor(std::ifstream& in, std::string& name, Tensor<float>& tensor) {
    // Read name
    uint32_t name_len;
    if (!in.read(reinterpret_cast<char*>(&name_len), sizeof(name_len))) {
        return false;
    }
    name.resize(name_len);
    in.read(&name[0], name_len);

    // Read number of dimensions
    uint32_t ndim;
    in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

    // Read shape
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        uint64_t dim;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        shape[i] = static_cast<size_t>(dim);
    }

    // Read data
    uint64_t numel;
    in.read(reinterpret_cast<char*>(&numel), sizeof(numel));

    tensor = Tensor<float>(shape);
    in.read(reinterpret_cast<char*>(tensor.data()), numel * sizeof(float));

    return in.good();
}

// Write string to stream
inline void write_string(std::ofstream& out, const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(str.data(), len);
}

// Read string from stream
inline std::string read_string(std::ifstream& in) {
    uint32_t len;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string str(len, '\0');
    in.read(&str[0], len);
    return str;
}

}  // namespace detail

// Save checkpoint to file
inline void save_checkpoint(
    const std::string& path,
    const nn::Module& model,
    const optim::Optimizer& optimizer,
    int epoch,
    int step,
    float loss,
    const std::string& config = "") {

    // Write to temp file first for atomic save
    std::string temp_path = path + ".tmp";
    std::ofstream out(temp_path, std::ios::binary);

    if (!out.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file: " + temp_path);
    }

    // Write header
    out.write(reinterpret_cast<const char*>(&detail::CKPT_MAGIC), sizeof(detail::CKPT_MAGIC));
    out.write(reinterpret_cast<const char*>(&detail::CKPT_VERSION), sizeof(detail::CKPT_VERSION));

    // Write metadata
    out.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
    out.write(reinterpret_cast<const char*>(&step), sizeof(step));
    out.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
    detail::write_string(out, config);

    // Write model state
    auto model_state = model.state_dict();
    uint32_t num_model_tensors = static_cast<uint32_t>(model_state.size());
    out.write(reinterpret_cast<const char*>(&num_model_tensors), sizeof(num_model_tensors));
    for (const auto& kv : model_state) {
        detail::write_tensor(out, kv.first, kv.second);
    }

    // Write optimizer state
    auto opt_state = optimizer.state_dict();
    uint32_t num_opt_tensors = static_cast<uint32_t>(opt_state.size());
    out.write(reinterpret_cast<const char*>(&num_opt_tensors), sizeof(num_opt_tensors));
    for (const auto& kv : opt_state) {
        detail::write_tensor(out, kv.first, kv.second);
    }

    out.close();

    // Atomic rename
    std::rename(temp_path.c_str(), path.c_str());
}

// Load checkpoint from file
inline Checkpoint load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary);

    if (!in.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file: " + path);
    }

    Checkpoint ckpt;

    // Read and verify header
    uint32_t magic, version;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != detail::CKPT_MAGIC) {
        throw std::runtime_error("Invalid checkpoint file (bad magic number)");
    }
    if (version != detail::CKPT_VERSION) {
        throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version));
    }

    // Read metadata
    in.read(reinterpret_cast<char*>(&ckpt.epoch), sizeof(ckpt.epoch));
    in.read(reinterpret_cast<char*>(&ckpt.step), sizeof(ckpt.step));
    in.read(reinterpret_cast<char*>(&ckpt.loss), sizeof(ckpt.loss));
    ckpt.config_json = detail::read_string(in);

    // Read model state
    uint32_t num_model_tensors;
    in.read(reinterpret_cast<char*>(&num_model_tensors), sizeof(num_model_tensors));
    for (uint32_t i = 0; i < num_model_tensors; ++i) {
        std::string name;
        Tensor<float> tensor;
        if (!detail::read_tensor(in, name, tensor)) {
            throw std::runtime_error("Failed to read model tensor at index " + std::to_string(i));
        }
        ckpt.model_state[name] = std::move(tensor);
    }

    // Read optimizer state
    uint32_t num_opt_tensors;
    in.read(reinterpret_cast<char*>(&num_opt_tensors), sizeof(num_opt_tensors));
    for (uint32_t i = 0; i < num_opt_tensors; ++i) {
        std::string name;
        Tensor<float> tensor;
        if (!detail::read_tensor(in, name, tensor)) {
            throw std::runtime_error("Failed to read optimizer tensor at index " + std::to_string(i));
        }
        ckpt.optimizer_state[name] = std::move(tensor);
    }

    return ckpt;
}

// Apply checkpoint to model and optimizer
inline void restore_checkpoint(
    const Checkpoint& ckpt,
    nn::Module& model,
    optim::Optimizer& optimizer) {

    model.load_state_dict(ckpt.model_state);
    optimizer.load_state_dict(ckpt.optimizer_state);
}

// Convenience function to check if checkpoint file exists
inline bool checkpoint_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

}  // namespace lightwatch
