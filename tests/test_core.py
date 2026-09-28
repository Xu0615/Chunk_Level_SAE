from __future__ import annotations

import concurrent.futures
import tempfile
import json
import random
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import chunk_saes.sae as sae_module
from chunk_saes.data import (
    ChunkPair,
    assign_token_sample,
    deterministic_token_locations,
    iter_documents,
    sample_adjacent_pair,
)
from chunk_saes.modeling import LayerBatch
from chunk_saes.rank_shards import (
    iter_rank_document_shard,
    load_rank_shard_manifest,
    prepare_rank_document_shards,
)
from chunk_saes.sae import (
    BatchTopKSAE,
    JointChunkSAE,
    learning_rate_multiplier,
    load_sae,
    save_sae,
)
from chunk_saes.sae import SAE_PARAMETER_SCHEMA_VERSION
from chunk_saes.frozen_sae import (
    selected_decoder_target_view,
    selected_joint_decoder_vectors,
)
from chunk_saes.utils import PILE_SOURCES, document_split
from extract_feature_evidence import FeatureScorer
from extract_probe_features import EncoderOnly
from run_linear_probes import csr_from_arrays


class _ThreadedCollectives:
    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.local = threading.local()
        self.condition = threading.Condition()
        self.slots: dict[int, dict] = {}

    def set_rank(self, rank: int) -> None:
        self.local.rank = rank
        self.local.sequence = 0

    def is_initialized(self) -> bool:
        return True

    def get_world_size(self) -> int:
        return self.world_size

    def get_rank(self) -> int:
        return self.local.rank

    def _slot(self) -> tuple[int, dict]:
        sequence = self.local.sequence
        self.local.sequence += 1
        return sequence, self.slots.setdefault(
            sequence,
            {"items": {}, "done": False, "remaining": self.world_size},
        )

    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        rank = self.get_rank()
        with self.condition:
            sequence, slot = self._slot()
            slot["items"][rank] = (output, value.clone())
            if len(slot["items"]) == self.world_size:
                gathered = torch.cat(
                    [slot["items"][index][1] for index in range(self.world_size)]
                )
                for target, _ in slot["items"].values():
                    target.copy_(gathered)
                slot["done"] = True
                self.condition.notify_all()
            else:
                assert self.condition.wait_for(
                    lambda: slot["done"], timeout=5.0
                ), f"collective {sequence} timed out"
            slot["remaining"] -= 1
            if slot["remaining"] == 0:
                self.slots.pop(sequence, None)

    def all_reduce(self, tensor: torch.Tensor, op=None) -> None:
        rank = self.get_rank()
        with self.condition:
            sequence, slot = self._slot()
            slot["items"][rank] = tensor
            if len(slot["items"]) == self.world_size:
                values = [
                    slot["items"][index].clone()
                    for index in range(self.world_size)
                ]
                if op == self.ReduceOp.MAX:
                    reduced = torch.stack(values).max(dim=0).values
                else:
                    reduced = torch.stack(values).sum(dim=0)
                for target in slot["items"].values():
                    target.copy_(reduced)
                slot["done"] = True
                self.condition.notify_all()
            else:
                assert self.condition.wait_for(
                    lambda: slot["done"], timeout=5.0
                ), f"collective {sequence} timed out"
            slot["remaining"] -= 1
            if slot["remaining"] == 0:
                self.slots.pop(sequence, None)


class FakeTokenizer:
    @staticmethod
    def decode(ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def test_adjacent_chunks_are_exact_and_non_overlapping():
    generator = torch.Generator().manual_seed(0)
    pair = sample_adjacent_pair(
        list(range(2048)),
        FakeTokenizer(),
        lengths=[32, 64, 128, 256, 512],
        generator=generator,
        doc_id="doc",
        source="Pile-CC",
        len_a=64,
        len_b=256,
    )
    assert pair is not None
    assert len(pair.input_ids_a) == 64
    assert len(pair.input_ids_b) == 256
    assert pair.input_ids_a[-1] + 1 == pair.input_ids_b[0]


def test_document_split_is_stable_and_document_level():
    assert document_split("same-document", 42) == document_split("same-document", 42)
    assert len(PILE_SOURCES) == 17


def test_batch_topk_and_checkpoint_roundtrip():
    model = BatchTopKSAE(8, 32, 3)
    inputs = torch.randn(4, 8)
    reconstructed, features, _, row_counts, feature_counts = model(
        inputs,
        batch_topk=True,
        return_activity_counts=True,
    )
    assert reconstructed.shape == inputs.shape
    assert int((features != 0).sum()) == 12
    assert torch.equal(row_counts, (features != 0).sum(dim=1))
    assert torch.equal(feature_counts, (features != 0).sum(dim=0))
    with tempfile.TemporaryDirectory() as directory:
        save_sae(model, directory, {"activation_dim": 8, "dict_size": 32, "k": 3})
        loaded = load_sae(directory).model
        loaded_reconstructed, _, _ = loaded(inputs, batch_topk=True)
        torch.testing.assert_close(loaded_reconstructed, reconstructed)


def test_joint_decoder_heads_are_explicit_and_roundtrip(tmp_path: Path):
    model = JointChunkSAE(4, 8, 2, decoder_backend="dense")
    with torch.no_grad():
        model.decoder_weight.copy_(torch.arange(32.0).reshape(4, 8))
        model.decoder_cross_weight.copy_(
            torch.arange(32.0, 64.0).reshape(4, 8)
        )
        model.decoder_bias.fill_(1.0)
        model.decoder_cross_bias.fill_(2.0)

    features = torch.zeros(1, 8)
    features[0, 3] = 2.0
    assert model.decoder_head_names == ("mean", "cross")
    assert model.decoder_mean_weight is model.decoder_weight
    assert model.decoder_mean_bias is model.decoder_bias
    mean = model.decode(features, head="mean")
    cross = model.decode(features, head="cross")
    torch.testing.assert_close(
        mean.squeeze(0), 2.0 * model.decoder_weight[:, 3] + 1.0
    )
    torch.testing.assert_close(
        cross.squeeze(0),
        2.0 * model.decoder_cross_weight[:, 3] + 2.0,
    )
    torch.testing.assert_close(model.decode_mean(features), mean)
    torch.testing.assert_close(model.decode_cross(features), cross)
    torch.testing.assert_close(
        model.decoder_vector(3, head="mean"), model.decoder_weight[:, 3]
    )
    torch.testing.assert_close(
        model.decoder_vector(3, head="cross"), model.decoder_cross_weight[:, 3]
    )
    with pytest.raises(ValueError, match="mean.*cross"):
        model.decoder_parameters("invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires head"):
        model.decoder_vector(3)

    checkpoint = tmp_path / "joint"
    save_sae(
        model,
        checkpoint,
        {
            "activation_dim": 4,
            "dict_size": 8,
            "k": 2,
            "decoder_heads": 2,
            "joint_chunk_alpha": 0.25,
            "joint_chunk_mean_baseline": 4.0,
            "joint_chunk_cross_baseline": 9.0,
        },
    )
    loaded = load_sae(checkpoint).model
    assert isinstance(loaded, JointChunkSAE)
    torch.testing.assert_close(loaded.decode(features, head="cross"), cross)
    with pytest.raises(ValueError, match="two decoder heads"):
        selected_decoder_target_view(checkpoint, [3])
    vectors, bias, _scale = selected_decoder_target_view(
        checkpoint,
        [3],
        head="cross",
    )
    torch.testing.assert_close(vectors[0], model.decoder_cross_weight[:, 3])
    torch.testing.assert_close(bias, model.decoder_cross_bias)
    joint = selected_joint_decoder_vectors(checkpoint, [3])
    mean_factor = 1.0 / ((1.25 * 4.0) ** 0.5)
    cross_factor = 0.5 / ((1.25 * 9.0) ** 0.5)
    expected_joint = torch.cat(
        (
            model.decoder_weight[:, 3] * mean_factor,
            model.decoder_cross_weight[:, 3] * cross_factor,
        )
    )
    torch.testing.assert_close(joint[0], expected_joint)


def test_joint_nested_prefix_uses_one_decoder_and_independent_cross_bias(
    tmp_path: Path,
):
    model = JointChunkSAE(
        4,
        8,
        2,
        decoder_backend="dense",
        cross_prefix=4,
    )
    with torch.no_grad():
        model.decoder_weight.copy_(torch.arange(32.0).reshape(4, 8))
        model.decoder_bias.fill_(1.0)
        model.decoder_cross_bias.fill_(2.0)

    features = torch.zeros(1, 8)
    features[0, 1] = 3.0
    features[0, 6] = 5.0
    mean = model.decode_mean(features)
    cross = model.decode_cross(features)
    torch.testing.assert_close(
        mean.squeeze(0),
        3.0 * model.decoder_weight[:, 1]
        + 5.0 * model.decoder_weight[:, 6]
        + 1.0,
    )
    torch.testing.assert_close(
        cross.squeeze(0),
        3.0 * model.decoder_weight[:, 1] + 2.0,
    )
    assert model.decoder_cross_weight is None
    assert model.feature_role(1) == "shared"
    assert model.feature_role(6) == "self_only"
    torch.testing.assert_close(
        model.decoder_vector(1),
        model.decoder_weight[:, 1],
    )
    torch.testing.assert_close(
        model.decoder_vector(1, head="cross"),
        model.decoder_weight[:, 1],
    )
    with pytest.raises(ValueError, match="self_only"):
        model.decoder_vector(6, head="cross")

    checkpoint = tmp_path / "joint_nested"
    save_sae(
        model,
        checkpoint,
        {
            "activation_dim": 4,
            "dict_size": 8,
            "k": 2,
            "decoder_heads": 1,
            "joint_chunk_layout": "nested_prefix",
            "joint_cross_prefix": 4,
        },
    )
    loaded = load_sae(checkpoint).model
    assert isinstance(loaded, JointChunkSAE)
    assert loaded.cross_prefix_size == 4
    assert loaded.decoder_cross_weight is None
    torch.testing.assert_close(loaded.decode_mean(features), mean)
    torch.testing.assert_close(loaded.decode_cross(features), cross)
    vectors, bias, _scale = selected_decoder_target_view(
        checkpoint,
        [1],
        head="cross",
    )
    torch.testing.assert_close(vectors[0], model.decoder_weight[:, 1])
    torch.testing.assert_close(bias, model.decoder_cross_bias)
    with pytest.raises(ValueError, match="only feature ids"):
        selected_decoder_target_view(checkpoint, [6], head="cross")


def test_single_head_decoder_view_remains_unambiguous(tmp_path: Path):
    model = BatchTopKSAE(4, 8, 2)
    checkpoint = tmp_path / "mean"
    save_sae(
        model,
        checkpoint,
        {"activation_dim": 4, "dict_size": 8, "k": 2},
    )
    vectors, bias, scale = selected_decoder_target_view(checkpoint, [1, 5])
    torch.testing.assert_close(vectors, model.decoder_weight[:, [1, 5]].T)
    torch.testing.assert_close(bias, model.decoder_bias)
    assert scale == 1.0
    with pytest.raises(ValueError, match="no decoder head"):
        selected_decoder_target_view(checkpoint, [1], head="cross")


def test_distributed_batch_topk_adapts_without_changing_exact_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collectives = _ThreadedCollectives(world_size=2)
    monkeypatch.setattr(sae_module, "dist", collectives)

    def run_rank(rank: int) -> dict[str, torch.Tensor]:
        collectives.set_rank(rank)
        model = BatchTopKSAE(
            2,
            8,
            2,
            batch_topk_candidate_multiplier=1.0,
        )
        pre = (
            torch.arange(100.0, 76.0, -1.0).reshape(3, 8)
            if rank == 0
            else torch.arange(24.0, 0.0, -1.0).reshape(3, 8)
        )
        encoded, threshold, active, _ = model.batch_topk(
            pre,
            distributed=True,
            return_activity_counts=True,
        )
        return {
            "pre": pre,
            "encoded": encoded,
            "threshold": threshold,
            "active": active,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(run_rank, range(2)))
    flat = torch.cat([item["pre"].flatten() for item in outputs])
    budget = 2 * 3 * 2
    threshold = flat.topk(budget, sorted=False).values.min()
    mask = flat > threshold
    ties_needed = budget - int(mask.sum())
    if ties_needed:
        ties = torch.nonzero(flat == threshold, as_tuple=False).flatten()
        mask[ties[:ties_needed]] = True
    expected = (flat * mask).reshape(6, 8)
    observed = torch.cat([item["encoded"] for item in outputs])
    assert torch.equal(observed, expected)
    assert all(float(item["threshold"]) == float(threshold) for item in outputs)


def test_bf16_histogram_batch_topk_matches_fallback_with_ties_and_mask():
    pre = torch.tensor(
        [
            [4.0, 3.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 3.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    sample_mask = torch.tensor([True, True, False, True])
    histogram = BatchTopKSAE(2, 8, 3, batch_topk_bf16_histogram=True)
    fallback = BatchTopKSAE(2, 8, 3, batch_topk_bf16_histogram=False)
    observed = histogram.batch_topk(
        pre,
        distributed=False,
        return_activity_counts=True,
        sample_mask=sample_mask,
        global_sample_count=int(sample_mask.sum()),
    )
    expected = fallback.batch_topk(
        pre,
        distributed=False,
        return_activity_counts=True,
        sample_mask=sample_mask,
        global_sample_count=int(sample_mask.sum()),
    )
    for left, right in zip(observed, expected, strict=True):
        assert torch.equal(left, right)
    encoded, _threshold, active_per_sample, active_per_feature = observed
    assert int((encoded != 0).sum()) == 9
    assert int(active_per_sample[2]) == 0
    assert int(active_per_feature.sum()) == 9


def test_dead_feature_age_tracks_accepted_occurrences():
    model = BatchTopKSAE(2, 4, 1)
    model.update_dead_feature_stats_(
        torch.tensor([2, 0, 1, 0]),
        global_samples=32,
        distributed=False,
    )
    assert model.num_occurrences_since_fired.tolist() == [0, 32, 0, 32]
    model.update_dead_feature_stats_(
        torch.tensor([0, 0, 0, 1]),
        global_samples=24,
        distributed=False,
    )
    assert model.num_occurrences_since_fired.tolist() == [24, 56, 24, 0]
    assert model.dead_feature_count(50) == 1


def test_auxk_can_activate_before_formal_dead_threshold():
    model = BatchTopKSAE(2, 4, 1)
    model.num_occurrences_since_fired.copy_(
        torch.tensor([4_999_999, 5_000_000, 9_999_999, 10_000_000])
    )
    assert model.auxiliary_feature_ids(5_000_000, 4).tolist() == [1, 2, 3]
    assert model.dead_feature_count(10_000_000) == 1


def test_cosine_learning_rate_floor_preserves_warmup_and_late_rescue():
    assert learning_rate_multiplier(0, 1000, 100, 0.1) == pytest.approx(0.01)
    assert learning_rate_multiplier(99, 1000, 100, 0.1) == pytest.approx(1.0)
    assert learning_rate_multiplier(999, 1000, 100, 0.1) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="min_ratio"):
        learning_rate_multiplier(0, 1000, 100, 1.1)


def test_encoder_input_center_and_decoder_output_bias_are_independent():
    model = BatchTopKSAE(3, 8, 2)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    model.pre_bias.copy_(torch.tensor([0.5, 1.0, 1.5]))
    model.decoder_bias.data.copy_(torch.tensor([10.0, 20.0, 30.0]))
    expected = torch.nn.functional.linear(
        inputs - model.pre_bias,
        model.encoder_weight,
        model.encoder_bias,
    )
    torch.testing.assert_close(
        model.pre_activations(inputs, apply_relu=False), expected
    )
    before = model.pre_activations(inputs, apply_relu=False)
    model.decoder_bias.data.add_(100.0)
    torch.testing.assert_close(
        model.pre_activations(inputs, apply_relu=False), before
    )


def test_downstream_encoders_use_pre_bias_and_validate_new_schema(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "activation_dim": 2,
        "dict_size": 2,
        "k": 1,
        "sae_parameter_schema_version": SAE_PARAMETER_SCHEMA_VERSION,
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors = {
        "encoder_weight": torch.eye(2),
        "encoder_bias": torch.zeros(2),
        "decoder_bias": torch.tensor([100.0, 100.0]),
        "pre_bias": torch.tensor([1.0, 2.0]),
        "threshold": torch.tensor(0.0),
        "activation_scale": torch.tensor(1.0),
        "feature_counts": torch.ones(2, dtype=torch.int64),
    }
    save_file(tensors, checkpoint / "sae.safetensors")

    hidden = torch.tensor([[3.0, 5.0]])
    expected = torch.tensor([[2.0, 3.0]])
    scorer = FeatureScorer(checkpoint, sample_size=2, seed=0, device="cpu")
    encoder = EncoderOnly(checkpoint, device="cpu")
    torch.testing.assert_close(scorer.scores(hidden), expected)
    torch.testing.assert_close(encoder.dense_code(hidden), expected)

    save_file(
        {key: value for key, value in tensors.items() if key != "pre_bias"},
        checkpoint / "sae.safetensors",
    )
    with pytest.raises(ValueError, match="lacks pre_bias"):
        FeatureScorer(checkpoint, sample_size=2, seed=0, device="cpu")
    with pytest.raises(ValueError, match="lacks pre_bias"):
        EncoderOnly(checkpoint, device="cpu")


def test_temporal_feature_evidence_samples_full_dictionary(tmp_path: Path):
    checkpoint = tmp_path / "temporal-checkpoint"
    checkpoint.mkdir()
    config = {
        "activation_dim": 2,
        "dict_size": 8,
        "k": 1,
        "mode": "temporal",
        "temporal_high_level_features": 2,
        "sae_parameter_schema_version": SAE_PARAMETER_SCHEMA_VERSION,
    }
    (checkpoint / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    save_file(
        {
            "encoder_weight": torch.randn(8, 2),
            "encoder_bias": torch.zeros(8),
            "decoder_bias": torch.zeros(2),
            "pre_bias": torch.zeros(2),
            "threshold": torch.tensor(0.0),
            "activation_scale": torch.tensor(1.0),
            "feature_counts": torch.ones(8, dtype=torch.int64),
        },
        checkpoint / "sae.safetensors",
    )

    scorer = FeatureScorer(
        checkpoint,
        sample_size=8,
        seed=153,
        device="cpu",
    )

    assert scorer.feature_ids.tolist() == list(range(8))
    assert max(scorer.feature_ids.tolist()) >= config["temporal_high_level_features"]
    encoder = EncoderOnly(checkpoint, device="cpu")
    assert encoder.dictionary_width == 8
    assert encoder.dense_code(torch.ones(1, 2)).shape == (1, 8)


def test_auxk_residual_reaches_dead_encoder_and_decoder_features():
    model = BatchTopKSAE(3, 16, 2)
    model.num_occurrences_since_fired.fill_(100)
    candidate_ids = model.auxiliary_feature_ids(10, 8)
    inputs = torch.randn(6, 3)
    target = torch.randn(6, 3)
    (
        _reconstructed,
        _features,
        _threshold,
        _active_rows,
        _active_features,
        auxiliary_reconstruction,
    ) = model(
        inputs,
        batch_topk=True,
        return_activity_counts=True,
        auxiliary_feature_ids=candidate_ids,
        auxiliary_k=4,
        return_auxiliary=True,
    )
    (auxiliary_reconstruction - target).square().mean().backward()
    assert model.encoder_weight.grad is not None
    assert model.decoder_weight.grad is not None
    assert bool(
        (model.encoder_weight.grad[candidate_ids].abs().sum(dim=1) > 0).any()
    )
    assert bool(
        (model.decoder_weight.grad[:, candidate_ids].abs().sum(dim=0) > 0).any()
    )


def test_prepared_rank_shards_match_direct_document_iteration():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = root / "tiny.jsonl"
        records = [
            {"text": f"Document {index} has enough alphabetic content for filtering. " * 4,
             "meta": {"id": f"doc-{index}", "pile_set_name": "Pile-CC"}}
            for index in range(12)
        ]
        records.append(records[3])
        with dataset.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        shard_root = root / "shards"
        prepare_rank_document_shards(
            dataset,
            shard_root,
            split_seed=42,
            split="train",
            world_size=2,
            excluded_sources={"ArXiv"},
            min_chars=1,
        )
        manifest = load_rank_shard_manifest(
            shard_root,
            dataset,
            split_seed=42,
            split="train",
            world_size=2,
            excluded_sources={"ArXiv"},
            min_chars=1,
        )
        for rank in range(2):
            direct = list(
                iter_documents(
                    dataset,
                    split_seed=42,
                    wanted_split="train",
                    rank=rank,
                    world_size=2,
                    excluded_sources={"ArXiv"},
                    min_chars=1,
                )
            )
            prepared = list(iter_rank_document_shard(shard_root, rank, manifest))
            assert prepared == direct


def test_token_locations_are_deterministic_and_order_independent():
    pairs = []
    for index in range(4):
        pair = sample_adjacent_pair(
            list(range(128)),
            FakeTokenizer(),
            lengths=[16],
            generator=torch.Generator().manual_seed(index),
            doc_id=f"doc-{index}",
            source="Pile-CC",
            len_a=16,
            len_b=16,
        )
        assert pair is not None
        pairs.append(
            assign_token_sample(
                pair,
                sample_index=index,
                sample_seed=43,
                rank=2,
            )
        )
    expected = deterministic_token_locations(pairs)
    permutation = [2, 0, 3, 1]
    shuffled = deterministic_token_locations([pairs[index] for index in permutation])
    for shuffled_row, original_row in enumerate(permutation):
        assert shuffled[0][shuffled_row] == expected[0][original_row]
        assert shuffled[1][shuffled_row] == expected[1][original_row]


def test_pair_sampling_and_cache_order_are_forward_batch_size_invariant():
    def simulate(forward_batch_size: int):
        pair_generator = torch.Generator().manual_seed(43)
        queues = {16: [], 32: []}
        sampled = []
        processed = []
        token_ids = list(range(512))
        for sample_index in range(20):
            length = 16 if sample_index % 3 else 32
            pair = sample_adjacent_pair(
                token_ids,
                FakeTokenizer(),
                lengths=[16, 32],
                generator=pair_generator,
                doc_id=f"doc-{sample_index // 2}",
                source="Pile-CC",
                len_a=length,
                len_b=length,
                decode_text=False,
            )
            assert pair is not None
            assign_token_sample(pair, sample_index=sample_index, sample_seed=43, rank=0)
            sampled.append(
                (
                    pair.input_ids_a,
                    pair.input_ids_b,
                    pair.token_seed,
                )
            )
            queues[length].append(pair)
            if len(queues[length]) >= forward_batch_size:
                processed.extend(queues[length])
                queues[length] = []
        for pairs in queues.values():
            processed.extend(pairs)

        random.Random(17).shuffle(processed)
        by_index = sorted(processed, key=lambda pair: pair.sample_index)
        choices, indices = deterministic_token_locations(by_index)
        cache_rows = [
            (
                pair.sample_index,
                pair.input_ids_a,
                pair.input_ids_b,
                bool(choices[row]),
                int(indices[row]),
            )
            for row, pair in enumerate(by_index)
        ]
        return sampled, cache_rows

    assert simulate(2) == simulate(4) == simulate(8)


def test_sparse_probe_matrix_roundtrip():
    indices = np.asarray([[1, 3, -1], [0, 2, 4]], dtype=np.int32)
    values = np.asarray([[2.0, 1.0, 0.0], [1.0, 3.0, 4.0]], dtype=np.float16)
    nnz = np.asarray([2, 3], dtype=np.int16)
    matrix = csr_from_arrays(indices, values, nnz, dict_size=5)
    np.testing.assert_allclose(matrix.toarray(), [[0, 2, 0, 1, 0], [1, 0, 3, 0, 4]])
    capped = csr_from_arrays(indices, values, nnz, dict_size=5, max_nnz=2)
    np.testing.assert_allclose(capped.toarray(), [[0, 2, 0, 1, 0], [1, 0, 3, 0, 0]])
