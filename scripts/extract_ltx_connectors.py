"""Extract embeddings connector + text projection from GGUF into a combined safetensors checkpoint.

Creates a checkpoint file for LTXVGemmaCLIPModelLoader that includes both
the text projection weights and the embeddings connector weights.

Usage:
    cd /home/ubuntu/Documents/programs/img/comfyui
    .venv/bin/python custom_nodes/comfyui-pose-director/scripts/extract_ltx_connectors.py
"""

import os
import struct
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from gguf import GGUFReader
from gguf.quants import dequantize as gguf_dequantize
import gguf

# Paths
MODELS_BASE = "/home/ubuntu/Documents/programs/shared-docker-infra/models/image-gen/comfyui"
GGUF_PATH = os.path.join(MODELS_BASE, "unet/ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf")
TP_PATH = os.path.join(MODELS_BASE, "text_encoders/ltx-2.3_text_projection_bf16.safetensors")
OUTPUT_PATH = os.path.join(MODELS_BASE, "checkpoints/ltx-2.3-connectors-textproj.safetensors")

# GGML types that are directly representable as torch tensors (no dequant needed)
TORCH_COMPAT_TYPES = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

# Detection key needed so LTXVGemmaCLIPModelLoader recognizes this as an AV model
DETECTION_KEYS = {"audio_adaln_single.linear.weight"}


def dequant_tensor(tensor):
    """Dequantize a GGUF tensor to bf16, handling F32/F16 natively.

    GGUF stores tensor shapes in reverse (column-major) order compared to
    PyTorch's row-major convention, so we reverse the shape.
    """
    qtype = tensor.tensor_type
    # GGUF shape is column-major; PyTorch expects row-major
    shape = tuple(reversed(tensor.shape))

    if qtype in TORCH_COMPAT_TYPES:
        # F32/F16: read raw bytes directly into numpy with correct dtype
        if qtype == gguf.GGMLQuantizationType.F32:
            np_dtype = np.float32
        else:
            np_dtype = np.float16
        raw_bytes = bytes(tensor.data)
        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)
        return torch.from_numpy(arr.copy()).to(torch.bfloat16)
    else:
        # Quantized: use gguf dequantize, then reshape to PyTorch order
        data = np.array(tensor.data, dtype=np.uint8)
        dequantized = gguf_dequantize(data, qtype)
        return torch.from_numpy(dequantized).reshape(shape).to(torch.bfloat16)


print(f"Loading GGUF: {GGUF_PATH}")
reader = GGUFReader(GGUF_PATH)

# Extract config metadata
config_str = None
for field_name, field in reader.fields.items():
    if field_name == "config":
        config_str = bytes(field.parts[-1]).decode("utf-8")
        break

# Dequantize connector + detection tensors
print("Dequantizing connector + detection tensors to bf16...")
connector_sd = {}
for tensor in reader.tensors:
    name = tensor.name
    is_connector = "embeddings_connector" in name
    is_detection = name in DETECTION_KEYS
    if not (is_connector or is_detection):
        continue

    dequantized = dequant_tensor(tensor)
    connector_sd[f"model.diffusion_model.{name}"] = dequantized
    tag = "connector" if is_connector else "detection"
    print(f"  [{tag}] {name}: {tensor.tensor_type} -> {dequantized.shape} {dequantized.dtype}")

print(f"\nConnector tensors: {len(connector_sd)}")

# Verify key shapes
print("\nVerifying video connector learnable_registers...")
vr_key = "model.diffusion_model.video_embeddings_connector.learnable_registers"
ar_key = "model.diffusion_model.audio_embeddings_connector.learnable_registers"
if vr_key in connector_sd:
    print(f"  video: {connector_sd[vr_key].shape}")
if ar_key in connector_sd:
    print(f"  audio: {connector_sd[ar_key].shape}")

print(f"\nLoading text projection: {TP_PATH}")
tp_sd = load_file(TP_PATH)
print(f"Text projection tensors: {len(tp_sd)}")
for k in tp_sd:
    print(f"  {k}: {tp_sd[k].shape} {tp_sd[k].dtype}")

# Merge
combined_sd = {}
combined_sd.update(connector_sd)
combined_sd.update(tp_sd)

metadata = {}
if config_str:
    metadata["config"] = config_str

print(f"\nSaving combined checkpoint: {OUTPUT_PATH}")
print(f"Total keys: {len(combined_sd)}")
save_file(combined_sd, OUTPUT_PATH, metadata=metadata)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"Done! File size: {size_mb:.1f} MB")
