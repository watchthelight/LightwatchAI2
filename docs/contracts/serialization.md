# Serialization Contract

## Overview

This document specifies the weight serialization format for LightwatchAI2, ensuring compatibility with HuggingFace GPT-2 weights.

**Defined by:** Phase 37
**Consumers:** Phase 38 (CLI), Phase 40 (Integration)

## File Format

### Primary Format: Custom Binary (.lwt)

LightwatchAI2 uses a custom binary format optimized for direct memory mapping.

```
+------------------+
| Magic (8 bytes)  |  "LWATCHV1"
+------------------+
| Header (JSON)    |  Length-prefixed JSON metadata
+------------------+
| Tensor Index     |  Offset table for each tensor
+------------------+
| Tensor Data      |  Contiguous fp32 data, 64-byte aligned
+------------------+
```

### Header JSON Schema

```json
{
    "version": 1,
    "model_type": "gpt2",
    "config": {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "vocab_size": 50257,
        "block_size": 1024
    },
    "dtype": "float32",
    "byte_order": "little",
    "tensor_count": 148,
    "total_bytes": 497025024
}
```

### Tensor Index Entry

```cpp
struct TensorIndexEntry {
    char name[128];      // Null-terminated tensor name
    uint32_t ndim;       // Number of dimensions
    uint64_t shape[8];   // Shape (max 8 dims, unused = 0)
    uint64_t offset;     // Byte offset from start of tensor data section
    uint64_t size_bytes; // Size in bytes
};
```

## HuggingFace Compatibility

### Weight Loading from HuggingFace

LightwatchAI2 can load weights from HuggingFace `pytorch_model.bin` or `model.safetensors`.

#### Supported Source Formats

| Format | File | Priority | Notes |
|--------|------|----------|-------|
| SafeTensors | `model.safetensors` | 1 (preferred) | No pickle, safer |
| PyTorch | `pytorch_model.bin` | 2 | Requires pickle parsing |

#### Tensor Name Mapping

HuggingFace GPT-2 uses different naming conventions. Map as follows:

| HuggingFace Name | LightwatchAI2 Name | Shape |
|------------------|-------------------|-------|
| `wte.weight` | `transformer.wte.weight` | [50257, 768] |
| `wpe.weight` | `transformer.wpe.weight` | [1024, 768] |
| `h.{i}.ln_1.weight` | `transformer.h.{i}.ln_1.weight` | [768] |
| `h.{i}.ln_1.bias` | `transformer.h.{i}.ln_1.bias` | [768] |
| `h.{i}.attn.c_attn.weight` | `transformer.h.{i}.attn.c_attn.weight` | [768, 2304] |
| `h.{i}.attn.c_attn.bias` | `transformer.h.{i}.attn.c_attn.bias` | [2304] |
| `h.{i}.attn.c_proj.weight` | `transformer.h.{i}.attn.c_proj.weight` | [768, 768] |
| `h.{i}.attn.c_proj.bias` | `transformer.h.{i}.attn.c_proj.bias` | [768] |
| `h.{i}.ln_2.weight` | `transformer.h.{i}.ln_2.weight` | [768] |
| `h.{i}.ln_2.bias` | `transformer.h.{i}.ln_2.bias` | [768] |
| `h.{i}.mlp.c_fc.weight` | `transformer.h.{i}.mlp.c_fc.weight` | [768, 3072] |
| `h.{i}.mlp.c_fc.bias` | `transformer.h.{i}.mlp.c_fc.bias` | [3072] |
| `h.{i}.mlp.c_proj.weight` | `transformer.h.{i}.mlp.c_proj.weight` | [3072, 768] |
| `h.{i}.mlp.c_proj.bias` | `transformer.h.{i}.mlp.c_proj.bias` | [768] |
| `ln_f.weight` | `transformer.ln_f.weight` | [768] |
| `ln_f.bias` | `transformer.ln_f.bias` | [768] |

**Note:** `{i}` ranges from 0 to 11 for GPT-2 Small (12 layers).

### Weight Transposition

HuggingFace GPT-2 stores Conv1D weights as `[in_features, out_features]`.
LightwatchAI2 uses Linear layers with `[out_features, in_features]`.

**Transpose required for:**
- `attn.c_attn.weight`
- `attn.c_proj.weight`
- `mlp.c_fc.weight`
- `mlp.c_proj.weight`

### Tied Embeddings

GPT-2 ties input embeddings (`wte`) with the output projection (`lm_head`).

```cpp
// lm_head shares weights with wte
model.lm_head.weight = model.transformer.wte.weight;  // Same memory
```

## Data Types

| Type | Size | Format |
|------|------|--------|
| float32 | 4 bytes | IEEE 754 single precision |

**Byte order:** Little-endian (x86-64 native)

## Validation

### Checksum

Each `.lwt` file includes a CRC32 checksum of the tensor data section:

```cpp
uint32_t compute_checksum(const void* data, size_t size);
```

### Parameter Count Validation

For GPT-2 Small, total parameter count must be:

```
Embeddings:      50257 * 768 + 1024 * 768 = 39,387,648
Per layer:       768*2304 + 2304 + 768*768 + 768 + 768 + 768 +
                 768*3072 + 3072 + 3072*768 + 768 = 7,087,872
12 layers:       7,087,872 * 12 = 85,054,464
Final LN:        768 + 768 = 1,536
Total:           124,443,648 parameters (~124M)
```

## API

```cpp
namespace lightwatch::serialization {

// Save model to .lwt format
void save(const GPT2& model, const std::string& path);

// Load model from .lwt format
GPT2 load(const std::string& path);

// Load from HuggingFace format (auto-detects safetensors vs pytorch)
GPT2 load_huggingface(const std::string& directory);

// Convert HuggingFace to .lwt
void convert_huggingface(const std::string& hf_dir, const std::string& output_path);

}  // namespace lightwatch::serialization
```

## File Extension

| Extension | Format |
|-----------|--------|
| `.lwt` | LightwatchAI2 native format |
| `.safetensors` | HuggingFace SafeTensors (read-only) |
| `.bin` | PyTorch pickle (read-only) |
