from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
from transformers.utils import import_utils as transformers_import_utils

# The installed flash-linear-attention package blocks while importing its CUDA
# extensions on this cluster.  Qwen's reference PyTorch implementation is a
# supported fallback and is sufficient for the short, inference-only chunks
# used by the SAE pipeline.  Set CHUNK_SAES_ENABLE_FLA=1 on an environment
# where that extension is known to load correctly.
if os.environ.get("CHUNK_SAES_ENABLE_FLA") != "1":
    transformers_import_utils.is_flash_linear_attention_available = lambda: False

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
)

from .utils import dtype_from_name


@dataclass
class LayerBatch:
    hidden: torch.Tensor
    mask: torch.Tensor

    def means(self) -> torch.Tensor:
        weights = self.mask.to(self.hidden.dtype).unsqueeze(-1)
        return (self.hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


class TargetLayerExtractor:
    """Load a truncated Qwen text stack and return its final residual stream."""

    def __init__(
        self,
        model_path: str,
        layer: int,
        device: str,
        *,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        truncated_load: bool = True,
        offload_embeddings_to_cpu: bool = False,
    ) -> None:
        self.model_path = model_path
        self.layer = layer
        self.device = torch.device(device)
        self.dtype = dtype_from_name(dtype)
        self.offload_embeddings_to_cpu = (
            bool(offload_embeddings_to_cpu) and self.device.type == "cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.tokenizer.model_max_length = 10**12

        language_model = None
        if truncated_load:
            try:
                language_model = self._load_truncated_text_model(
                    model_path,
                    layer,
                    attn_implementation=attn_implementation,
                )
            except Exception as error:
                print(
                    "[target-layer] direct truncated load failed; "
                    f"falling back to full multimodal load: {error}",
                    flush=True,
                )
        if language_model is None:
            full_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=self.dtype,
                attn_implementation=attn_implementation,
                low_cpu_mem_usage=True,
            )
            language_model = full_model.model.language_model
            if not 0 <= layer < len(language_model.layers):
                raise ValueError(
                    f"layer={layer} outside [0, {len(language_model.layers)})"
                )
            language_model.layers = nn.ModuleList(
                list(language_model.layers[: layer + 1])
            )
            del full_model
        # The previous implementation captured the last decoder block with a
        # forward hook and then discarded the model's final RMSNorm output.
        # Replacing that norm with Identity makes last_hidden_state exactly the
        # same residual location without a hook or an unused normalization.
        language_model.norm = nn.Identity()
        if self.offload_embeddings_to_cpu:
            language_model.embed_tokens.to("cpu")
            language_model.rotary_emb.to(self.device)
            for decoder_layer in language_model.layers:
                decoder_layer.to(self.device)
            self.model = language_model.eval()
        else:
            self.model = language_model.to(self.device).eval()
        del language_model
        gc.collect()
        torch.cuda.empty_cache()
        self._all_ones_masks: dict[tuple[int, int], torch.Tensor] = {}

    def _load_truncated_text_model(
        self,
        model_path: str,
        layer: int,
        *,
        attn_implementation: str,
    ) -> Qwen3_5TextModel:
        root = Path(model_path)
        config = AutoConfig.from_pretrained(model_path)
        text_config = config.text_config
        total_layers = int(text_config.num_hidden_layers)
        if not 0 <= layer < total_layers:
            raise ValueError(f"layer={layer} outside [0, {total_layers})")
        text_config.num_hidden_layers = layer + 1
        text_config.layer_types = list(text_config.layer_types[: layer + 1])
        text_config._attn_implementation = attn_implementation
        with torch.device("meta"):
            model = Qwen3_5TextModel(text_config)
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("model safetensors index has no weight_map")
        selected_by_file: dict[str, list[tuple[str, str]]] = {}
        prefix = "model.language_model."
        for full_name, file_name in weight_map.items():
            if not full_name.startswith(prefix):
                continue
            short_name = full_name[len(prefix) :]
            if short_name.startswith("norm."):
                continue
            if short_name.startswith("layers."):
                layer_index = int(short_name.split(".", 2)[1])
                if layer_index > layer:
                    continue
            elif not (
                short_name.startswith("embed_tokens.")
                or short_name.startswith("rotary_emb.")
            ):
                continue
            selected_by_file.setdefault(str(file_name), []).append(
                (full_name, short_name)
            )
        state: dict[str, torch.Tensor] = {}
        for file_name, names in selected_by_file.items():
            with safe_open(
                str(root / file_name),
                framework="pt",
                device="cpu",
            ) as handle:
                for full_name, short_name in names:
                    state[short_name] = handle.get_tensor(full_name)
        missing, unexpected = model.load_state_dict(
            state,
            strict=False,
            assign=True,
        )
        meaningful_missing = [
            name for name in missing if not name.startswith("norm.")
        ]
        if meaningful_missing or unexpected:
            raise ValueError(
                "truncated text checkpoint mismatch: "
                f"missing={meaningful_missing[:10]}, unexpected={unexpected[:10]}"
            )
        # inv_freq/original_inv_freq are non-persistent buffers and therefore
        # absent from safetensors. A model created on the meta device leaves
        # them meta unless the tiny rotary module is materialized explicitly.
        model.rotary_emb = Qwen3_5TextRotaryEmbedding(
            config=text_config
        )
        return model

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def forward_ids(self, sequences: Sequence[Sequence[int]]) -> LayerBatch:
        if not sequences:
            raise ValueError("forward_ids requires at least one sequence")
        lengths = [len(ids) for ids in sequences]
        max_len = max(lengths)
        batch = len(sequences)
        pin_memory = self.device.type == "cuda"
        if min(lengths) == max_len:
            host_ids = torch.tensor(
                sequences,
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_memory,
            )
            mask_key = (batch, max_len)
            mask = self._all_ones_masks.get(mask_key)
            if mask is None or mask.device != self.device:
                mask = torch.ones(
                    mask_key,
                    dtype=torch.long,
                    device=self.device,
                )
                self._all_ones_masks[mask_key] = mask
        else:
            pad = int(self.tokenizer.pad_token_id)
            host_ids = torch.full(
                (batch, max_len),
                pad,
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_memory,
            )
            host_mask = torch.zeros(
                (batch, max_len),
                dtype=torch.long,
                device="cpu",
                pin_memory=pin_memory,
            )
            for row, ids in enumerate(sequences):
                length = lengths[row]
                host_ids[row, :length] = torch.as_tensor(ids, dtype=torch.long)
                host_mask[row, :length] = 1
            mask = host_mask.to(self.device, non_blocking=pin_memory)
        if self.offload_embeddings_to_cpu:
            inputs_embeds = self.model.embed_tokens(host_ids).to(
                self.device,
                non_blocking=pin_memory,
            )
            output = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
        else:
            input_ids = host_ids.to(self.device, non_blocking=pin_memory)
            output = self.model(
                input_ids=input_ids,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
        return LayerBatch(hidden=output.last_hidden_state, mask=mask)

    def close(self) -> None:
        self._all_ones_masks.clear()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def check_batch_invariance(
        self,
        sequences: Sequence[Sequence[int]],
        batch_sizes: Sequence[int],
    ) -> dict[str, object]:
        """Compare the same sequences under several batch shapes.

        This is intentionally a small preflight check. It reports exact BF16
        equality and a max absolute difference rather than silently treating a
        changed GEMM shape as bitwise identical.
        """

        sequences = list(sequences)
        if not sequences:
            raise ValueError("batch invariance check requires sequences")
        reference = None
        variants = []
        for batch_size in sorted(set(max(1, int(value)) for value in batch_sizes)):
            outputs = []
            for start in range(0, len(sequences), batch_size):
                current_sequences = list(sequences[start : start + batch_size])
                real_count = len(current_sequences)
                while len(current_sequences) < batch_size:
                    current_sequences.append(
                        sequences[(start + len(current_sequences)) % len(sequences)]
                    )
                batch = self.forward_ids(current_sequences)
                outputs.append(
                    torch.cat(
                        [
                            batch.hidden[index, : len(sequence)]
                            for index, sequence in enumerate(
                                current_sequences[:real_count]
                            )
                        ],
                        dim=0,
                    ).detach()
                )
            current = torch.cat(outputs, dim=0)
            digest = hashlib.sha256(
                memoryview(current.contiguous().view(torch.uint8).cpu().numpy())
            ).hexdigest()
            if reference is None:
                reference = current
                equal = True
                max_abs_diff = 0.0
                differing = 0
            else:
                equal = bool(torch.equal(reference, current))
                diff = (reference.float() - current.float()).abs()
                max_abs_diff = float(diff.max().item())
                differing = int((reference != current).sum().item())
            variants.append(
                {
                    "batch_size": batch_size,
                    "actual_batch_shape": batch_size,
                    "torch_equal": equal,
                    "max_abs_diff": max_abs_diff,
                    "differing_elements": differing,
                    "activation_digest": digest,
                }
            )
        strict = all(bool(item["torch_equal"]) for item in variants)
        return {
            "strict_bitwise": strict,
            "equivalence": "strict" if strict else "numerical",
            "sequences": len(sequences),
            "variants": variants,
        }
