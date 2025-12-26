#!/usr/bin/env python3
"""
Convert .lwbin format back to HuggingFace-compatible format.

Usage:
    python convert_lwbin_to_hf.py --input model.lwbin --output model_hf/
"""

import argparse
import struct
import numpy as np
from pathlib import Path

# .lwbin format constants
LWBIN_MAGIC = b'LWAI'
LWBIN_VERSION = 1
LWBIN_HEADER_SIZE = 64

# Dtype codes
DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1
DTYPE_INT32 = 2


def read_header(f):
    """Read and validate .lwbin header."""
    header = f.read(LWBIN_HEADER_SIZE)
    if len(header) != LWBIN_HEADER_SIZE:
        raise ValueError("Invalid file: header too short")

    magic = header[0:4]
    if magic != LWBIN_MAGIC:
        raise ValueError(f"Invalid magic: {magic}")

    version = struct.unpack_from('<I', header, 4)[0]
    if version > LWBIN_VERSION:
        raise ValueError(f"Unsupported version: {version}")

    tensor_count = struct.unpack_from('<I', header, 8)[0]
    return tensor_count


def read_tensor(f):
    """Read a single tensor from file."""
    # Name
    name_len = struct.unpack('<I', f.read(4))[0]
    name = f.read(name_len).decode('utf-8')

    # Dimensions
    ndims = struct.unpack('<I', f.read(4))[0]

    # Shape
    shape = []
    for _ in range(ndims):
        dim = struct.unpack('<q', f.read(8))[0]
        shape.append(dim)

    # Dtype
    dtype = struct.unpack('<B', f.read(1))[0]

    # Data size
    data_size = struct.unpack('<Q', f.read(8))[0]

    # Data
    data = f.read(data_size)

    if dtype == DTYPE_FLOAT32:
        tensor = np.frombuffer(data, dtype=np.float32).reshape(shape)
    elif dtype == DTYPE_FLOAT16:
        tensor = np.frombuffer(data, dtype=np.float16).reshape(shape)
    elif dtype == DTYPE_INT32:
        tensor = np.frombuffer(data, dtype=np.int32).reshape(shape)
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    return name, tensor


def convert_name_to_hf(lw_name):
    """Convert lightwatch parameter name to HuggingFace format."""
    name = lw_name

    # Handle embedding
    if name.startswith('embedding.wte.'):
        return 'transformer.' + name[len('embedding.'):]
    if name.startswith('embedding.wpe.'):
        return 'transformer.' + name[len('embedding.'):]

    # Handle layers
    if name.startswith('layer_'):
        parts = name.split('.')
        layer_idx = parts[0].split('_')[1]
        rest = '.'.join(parts[1:])

        # Convert attention names back
        rest = rest.replace('qkv_proj', 'c_attn')
        rest = rest.replace('out_proj', 'c_proj')

        # Convert MLP names back
        rest = rest.replace('fc1', 'c_fc')
        rest = rest.replace('fc2', 'c_proj')

        # Convert layer norm names
        rest = rest.replace('ln1', 'ln_1')
        rest = rest.replace('ln2', 'ln_2')

        return f'transformer.h.{layer_idx}.{rest}'

    # Handle final layer norm
    if name in ['ln_f.weight', 'ln_f.bias']:
        return 'transformer.' + name

    return name


def main():
    parser = argparse.ArgumentParser(description='Convert .lwbin to HuggingFace format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .lwbin file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for HuggingFace format')
    parser.add_argument('--verbose', action='store_true',
                        help='Print tensor names during conversion')
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("Error: torch library not installed.")
        print("Install with: pip install torch")
        return 1

    print(f"Reading: {args.input}")

    state_dict = {}
    with open(args.input, 'rb') as f:
        tensor_count = read_header(f)
        print(f"Found {tensor_count} tensors")

        for _ in range(tensor_count):
            lw_name, tensor = read_tensor(f)
            hf_name = convert_name_to_hf(lw_name)

            if args.verbose:
                print(f"  {lw_name} -> {hf_name}: {tensor.shape}")

            state_dict[hf_name] = torch.from_numpy(tensor.copy())

    # Save as PyTorch state dict
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, output_path / 'pytorch_model.bin')
    print(f"Saved to {output_path / 'pytorch_model.bin'}")

    return 0


if __name__ == '__main__':
    exit(main())
