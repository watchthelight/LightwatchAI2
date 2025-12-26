#!/usr/bin/env python3
"""
Convert HuggingFace GPT-2 weights to .lwbin format.

Usage:
    python convert_hf_to_lwbin.py --model gpt2 --output model.lwbin
    python convert_hf_to_lwbin.py --model gpt2-medium --output model.lwbin
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


def write_header(f, tensor_count):
    """Write .lwbin header."""
    header = bytearray(LWBIN_HEADER_SIZE)
    header[0:4] = LWBIN_MAGIC
    struct.pack_into('<I', header, 4, LWBIN_VERSION)
    struct.pack_into('<I', header, 8, tensor_count)
    f.write(header)


def write_tensor(f, name, tensor):
    """Write a single tensor to file."""
    # Name length and name
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)

    # Number of dimensions
    shape = tensor.shape
    f.write(struct.pack('<I', len(shape)))

    # Shape
    for dim in shape:
        f.write(struct.pack('<q', dim))  # int64

    # Dtype (always float32)
    f.write(struct.pack('<B', DTYPE_FLOAT32))

    # Data size
    data = tensor.astype(np.float32).tobytes()
    f.write(struct.pack('<Q', len(data)))  # uint64

    # Data
    f.write(data)


def convert_name(hf_name):
    """Convert HuggingFace parameter name to lightwatch format."""
    # HuggingFace GPT-2 naming: transformer.h.0.attn.c_attn.weight
    # Lightwatch naming: layer_0.attn.q_proj.weight

    name = hf_name

    # Handle transformer prefix
    if name.startswith('transformer.'):
        name = name[len('transformer.'):]

    # Handle wte/wpe
    if name.startswith('wte.'):
        return 'embedding.wte.' + name[4:]
    if name.startswith('wpe.'):
        return 'embedding.wpe.' + name[4:]

    # Handle layers
    if name.startswith('h.'):
        parts = name.split('.')
        layer_idx = parts[1]
        rest = '.'.join(parts[2:])

        # Convert attention names
        if rest.startswith('attn.'):
            rest = rest.replace('c_attn', 'qkv_proj')
            rest = rest.replace('c_proj', 'out_proj')

        # Convert MLP names
        if rest.startswith('mlp.'):
            rest = rest.replace('c_fc', 'fc1')
            rest = rest.replace('c_proj', 'fc2')

        # Convert layer norm names
        rest = rest.replace('ln_1', 'ln1')
        rest = rest.replace('ln_2', 'ln2')

        name = f'layer_{layer_idx}.{rest}'

    # Handle final layer norm
    if name == 'ln_f.weight' or name == 'ln_f.bias':
        return name

    return name


def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace GPT-2 to .lwbin')
    parser.add_argument('--model', type=str, default='gpt2',
                        help='HuggingFace model name (gpt2, gpt2-medium, etc.)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .lwbin file path')
    parser.add_argument('--verbose', action='store_true',
                        help='Print tensor names during conversion')
    args = parser.parse_args()

    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("Error: transformers library not installed.")
        print("Install with: pip install transformers")
        return 1

    print(f"Loading model: {args.model}")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    state_dict = model.state_dict()

    print(f"Converting {len(state_dict)} tensors...")

    with open(args.output, 'wb') as f:
        write_header(f, len(state_dict))

        for hf_name, tensor in state_dict.items():
            lw_name = convert_name(hf_name)
            tensor_np = tensor.cpu().numpy()

            if args.verbose:
                print(f"  {hf_name} -> {lw_name}: {tensor_np.shape}")

            write_tensor(f, lw_name, tensor_np)

    output_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Saved to {args.output} ({output_size:.1f} MB)")

    return 0


if __name__ == '__main__':
    exit(main())
