// Phase 08: Module Base Class (minimal version for embedding)
// Full implementation in Phase 11

#pragma once

#include <lightwatch/autograd.hpp>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace lightwatch::nn {

// Base class for neural network modules
class Module {
public:
    virtual ~Module() = default;

    // Forward pass - derived classes implement this
    virtual autograd::Variable forward(const autograd::Variable& input) = 0;

    // Multi-input forward (for attention, etc.)
    virtual autograd::Variable forward(
        const autograd::Variable& input,
        const autograd::Variable& other) {
        (void)other;
        return forward(input);
    }

    // Get all parameters
    std::vector<autograd::Variable*> parameters() {
        std::vector<autograd::Variable*> params;
        for (auto& p : parameters_) {
            params.push_back(p.second);
        }
        // Recursively get submodule parameters
        for (auto& m : submodules_) {
            auto sub_params = m.second->parameters();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        return params;
    }

    // Get named parameters
    std::vector<std::pair<std::string, autograd::Variable*>> named_parameters() {
        std::vector<std::pair<std::string, autograd::Variable*>> named;
        for (auto& p : parameters_) {
            named.push_back(p);
        }
        // Recursively get submodule parameters
        for (auto& m : submodules_) {
            auto sub_params = m.second->named_parameters();
            for (auto& sp : sub_params) {
                named.emplace_back(m.first + "." + sp.first, sp.second);
            }
        }
        return named;
    }

    // Count total parameters
    size_t num_parameters() const {
        size_t count = 0;
        for (const auto& p : parameters_) {
            count += p.second->numel();
        }
        for (const auto& m : submodules_) {
            count += m.second->num_parameters();
        }
        return count;
    }

    // Get all modules
    std::vector<Module*> modules() {
        std::vector<Module*> mods;
        mods.push_back(this);
        for (auto& m : submodules_) {
            auto sub_mods = m.second->modules();
            mods.insert(mods.end(), sub_mods.begin(), sub_mods.end());
        }
        return mods;
    }

    // Training mode
    void train(bool mode = true) {
        training_ = mode;
        for (auto& m : submodules_) {
            m.second->train(mode);
        }
    }

    void eval() { train(false); }

    bool is_training() const { return training_; }

    // Zero gradients
    void zero_grad() {
        for (auto& p : parameters_) {
            p.second->zero_grad();
        }
        for (auto& m : submodules_) {
            m.second->zero_grad();
        }
    }

    // Set requires_grad for all parameters
    void requires_grad_(bool requires_grad) {
        for (auto& p : parameters_) {
            p.second->set_requires_grad(requires_grad);
        }
        for (auto& m : submodules_) {
            m.second->requires_grad_(requires_grad);
        }
    }

    // State dict for serialization
    std::unordered_map<std::string, Tensor<float>> state_dict() const {
        std::unordered_map<std::string, Tensor<float>> dict;
        for (const auto& p : parameters_) {
            dict[p.first] = p.second->data();
        }
        for (const auto& m : submodules_) {
            auto sub_dict = m.second->state_dict();
            for (const auto& kv : sub_dict) {
                dict[m.first + "." + kv.first] = kv.second;
            }
        }
        return dict;
    }

    // Load state dict
    void load_state_dict(const std::unordered_map<std::string, Tensor<float>>& dict) {
        for (auto& p : parameters_) {
            auto it = dict.find(p.first);
            if (it != dict.end()) {
                p.second->data() = it->second;
            }
        }
        for (auto& m : submodules_) {
            std::unordered_map<std::string, Tensor<float>> sub_dict;
            std::string prefix = m.first + ".";
            for (const auto& kv : dict) {
                if (kv.first.substr(0, prefix.size()) == prefix) {
                    sub_dict[kv.first.substr(prefix.size())] = kv.second;
                }
            }
            m.second->load_state_dict(sub_dict);
        }
    }

protected:
    bool training_ = true;

    // Registration methods
    void register_parameter(const std::string& name, autograd::Variable& param) {
        parameters_.emplace_back(name, &param);
    }

    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        submodules_.emplace_back(name, module);
    }

    void register_buffer(const std::string& name, Tensor<float>& buffer) {
        buffers_.emplace_back(name, &buffer);
    }

private:
    std::vector<std::pair<std::string, autograd::Variable*>> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
    std::vector<std::pair<std::string, Tensor<float>*>> buffers_;
};

}  // namespace lightwatch::nn
