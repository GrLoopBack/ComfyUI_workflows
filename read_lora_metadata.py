#!/usr/bin/env python3
"""
read_lora_metadata.py
---------------------
Reads and displays metadata from a LoRA .safetensors file.
Extracts: base model/architecture, trigger words, training info,
tensor shapes, and any other embedded metadata.

Usage:
    python read_lora_metadata.py path/to/your/lora.safetensors

Requirements:
    pip install safetensors
"""

import sys
import json
import struct
import argparse
from pathlib import Path


def read_safetensors_header(filepath: str) -> dict:
    """
    Read the JSON header from a safetensors file without loading tensors.
    The format is: [8-byte LE uint64 header_size][header_size bytes of JSON][tensor data]
    """
    with open(filepath, "rb") as f:
        # First 8 bytes = length of the JSON header
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError("File too small to be a valid safetensors file.")
        header_len = struct.unpack("<Q", raw_len)[0]

        # Read the JSON header
        raw_header = f.read(header_len)
        if len(raw_header) < header_len:
            raise ValueError("Truncated safetensors header.")

    return json.loads(raw_header.decode("utf-8"))


def extract_metadata(header: dict) -> dict:
    """Pull out the __metadata__ block (if present) and tensor info."""
    metadata = header.get("__metadata__", {})
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return metadata, tensors


def guess_architecture(metadata: dict, tensor_keys: list) -> str:
    """
    Heuristic: infer the base architecture from metadata fields and tensor names.
    """
    hints = []

    # Common metadata keys that name the base model
    for key in ("ss_base_model_version", "ss_sd_model_name", "baseModel",
                "base_model", "modelspec.architecture", "ss_network_module"):
        val = metadata.get(key, "")
        if val:
            hints.append(f"{key} = {val}")

    # Tensor name patterns
    key_str = " ".join(tensor_keys)
    arch_hints = []
    if "lora_unet" in key_str:
        arch_hints.append("SD-style UNet (SD 1.x / SD 2.x / SDXL)")
    if "transformer." in key_str or "single_transformer" in key_str:
        arch_hints.append("DiT / Flux / SD3-style transformer")
    if "text_encoder" in key_str:
        arch_hints.append("includes text encoder weights")
    if "unet" in key_str.lower() and "sdxl" in key_str.lower():
        arch_hints.append("SDXL UNet")
    if arch_hints:
        hints.extend(arch_hints)

    return "\n    ".join(hints) if hints else "Unknown (no architecture metadata found)"


def pretty_print_metadata(metadata: dict, tensors: dict, filepath: str):
    sep = "─" * 70

    print(f"\n{sep}")
    print(f"  LoRA Metadata Inspector")
    print(f"  File: {Path(filepath).name}")
    print(sep)

    # ── Trigger words ──────────────────────────────────────────────────────
    print("\n🔑  TRIGGER WORDS / TAGS")
    trigger_keys = [
        "ss_tag_frequency", "trigger_words", "activation_text",
        "ss_caption_prefix", "instancePrompt", "instance_prompt",
        "modelspec.trigger_phrase",
    ]
    found_triggers = False
    for key in trigger_keys:
        val = metadata.get(key)
        if val:
            found_triggers = True
            print(f"  [{key}]")
            # ss_tag_frequency is JSON-encoded tag counts; pretty-print it
            if key == "ss_tag_frequency":
                try:
                    tag_data = json.loads(val) if isinstance(val, str) else val
                    for subset, tags in tag_data.items():
                        print(f"    Subset: {subset}")
                        # Sort tags by frequency descending
                        for tag, freq in sorted(tags.items(), key=lambda x: -x[1])[:30]:
                            print(f"      {freq:>5}x  {tag}")
                        if len(tags) > 30:
                            print(f"      … and {len(tags) - 30} more tags")
                except Exception:
                    print(f"    {val}")
            else:
                print(f"    {val}")
    if not found_triggers:
        print("  (none found — this LoRA may not require a trigger word)")

    # ── Base model / architecture ──────────────────────────────────────────
    print("\n🏗   BASE MODEL / ARCHITECTURE")
    arch = guess_architecture(metadata, list(tensors.keys()))
    print(f"    {arch}")

    # ── Training details ───────────────────────────────────────────────────
    print("\n📊  TRAINING DETAILS")
    training_keys = [
        "ss_learning_rate", "ss_unet_lr", "ss_text_encoder_lr",
        "ss_num_epochs", "ss_epoch", "ss_steps", "ss_batch_size_per_device",
        "ss_gradient_accumulation_steps", "ss_optimizer", "ss_lr_scheduler",
        "ss_network_dim", "ss_network_alpha", "ss_mixed_precision",
        "ss_clip_skip", "ss_max_token_length", "ss_seed",
        "ss_dataset_dirs", "ss_num_train_images", "ss_resolution",
        "ss_training_comment",
    ]
    found_any = False
    for key in training_keys:
        val = metadata.get(key)
        if val not in (None, "", "{}"):
            found_any = True
            label = key.replace("ss_", "").replace("_", " ").title()
            # Pretty-print nested JSON blobs
            try:
                parsed = json.loads(val) if isinstance(val, str) else val
                if isinstance(parsed, dict):
                    print(f"  {label}:")
                    for k, v in parsed.items():
                        print(f"    {k}: {v}")
                    continue
            except Exception:
                pass
            print(f"  {label}: {val}")
    if not found_any:
        print("  (no standard training metadata found)")

    # ── All raw metadata ───────────────────────────────────────────────────
    print("\n📋  ALL RAW METADATA KEYS")
    if metadata:
        for key, val in sorted(metadata.items()):
            if key in training_keys + trigger_keys:
                continue  # already shown above
            # Truncate very long values
            display = str(val)
            if len(display) > 120:
                display = display[:117] + "…"
            print(f"  {key}: {display}")
    else:
        print("  (file has no __metadata__ block)")

    # ── Tensor summary ─────────────────────────────────────────────────────
    print("\n🧮  TENSOR SUMMARY")
    print(f"  Total tensors: {len(tensors)}")
    if tensors:
        # Show a sample of tensor names and shapes
        sample = list(tensors.items())[:10]
        print("  First 10 tensors:")
        for name, info in sample:
            shape = info.get("shape", "?")
            dtype = info.get("dtype", "?")
            print(f"    {name:<60}  shape={shape}  dtype={dtype}")
        if len(tensors) > 10:
            print(f"    … and {len(tensors) - 10} more")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Display metadata from a LoRA .safetensors file."
    )
    parser.add_argument("filepath", help="Path to the .safetensors file")
    parser.add_argument(
        "--json", action="store_true",
        help="Dump raw __metadata__ block as formatted JSON and exit"
    )
    args = parser.parse_args()

    filepath = args.filepath
    if not Path(filepath).exists():
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading header from: {filepath}")
    header = read_safetensors_header(filepath)
    metadata, tensors = extract_metadata(header)

    if args.json:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    pretty_print_metadata(metadata, tensors, filepath)


if __name__ == "__main__":
    main()
