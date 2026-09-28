from __future__ import annotations

import copy
import sys
import types

import torch
import pytest

from chunk_saes.forward import ForwardSchedule, process_window_by_length
from chunk_saes.hf_corpus import (
    HFStreamingCorpus,
    iter_hf_documents,
    iter_hf_prepared_documents,
    verify_hf_stream,
)
from chunk_saes.modeling import LayerBatch
from chunk_saes.sae import BatchTopKSAE
from chunk_saes.sample_plan import PlanRow
from chunk_saes.streaming_dispatch import deserialize_rows, serialize_rows
from chunk_saes.tensor_parallel import (
    FeatureTensorParallelSAE,
    load_sharded_training_checkpoint,
    save_sharded_training_checkpoint,
)
from chunk_saes.sae import (
    SAE_PARAMETER_SCHEMA_VERSION,
    SAE_TRAINING_IMPLEMENTATION_VERSION,
)
from train_chunk_saes import chunk_input_deduplication


class _FakeIterableDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def shuffle(self, seed=None, buffer_size=1000):
        return self

    def shard(self, num_shards, index, contiguous=False):
        return _FakeIterableDataset(self.rows[index::num_shards])

    def __iter__(self):
        return iter(self.rows)


def row(pair_id: int, a: tuple[int, ...], b: tuple[int, ...]) -> PlanRow:
    return PlanRow(
        pair_id=pair_id,
        occurrence_start=sum(
            (len((1, 2)) + len((3, 4, 5)),)
        )
        * pair_id,
        doc_hash=bytes([pair_id + 1]) * 32,
        content_hash=bytes([pair_id + 2]) * 32,
        source="source",
        input_ids_a=a,
        input_ids_b=b,
        start_a=0,
        start_b=len(a),
        document_token_count=len(a) + len(b),
        execution_rank=0,
    )


def test_streaming_dispatch_roundtrip() -> None:
    rows = [
        row(0, (1, 2), (3, 4, 5)),
        row(1, (6,), (7, 8)),
    ]
    payload = serialize_rows(rows, source_to_id={"source": 0})
    restored = deserialize_rows(payload, id_to_source=["source"])
    assert restored == rows


def test_hf_deduplicated_stream_split_and_rank_sharding(
    monkeypatch,
) -> None:
    rows = [{"text": f"deduplicated document {index}"} for index in range(100)]
    fake_module = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: _FakeIterableDataset(rows)
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    spec = HFStreamingCorpus(
        dataset="EleutherAI/the_pile_deduplicated",
        logical_split="train",
        split_seed=42,
        shuffle_buffer=16,
    )
    all_documents = list(iter_hf_documents(spec))
    rank_zero = list(iter_hf_documents(spec, rank=0, world_size=2))
    rank_one = list(iter_hf_documents(spec, rank=1, world_size=2))
    assert all(document.source == "Pile-Deduplicated" for document in all_documents)
    assert {
        document.doc_id for document in rank_zero + rank_one
    } == {document.doc_id for document in all_documents}
    assert not (
        {document.doc_id for document in rank_zero}
        & {document.doc_id for document in rank_one}
    )
    prepared = list(iter_hf_prepared_documents(spec))
    assert [item.doc_id for item in prepared] == [
        item.doc_id for item in all_documents
    ]
    verified = verify_hf_stream(spec)
    assert verified["verified"] is True


class _Extractor:
    hidden_size = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def forward_ids(self, sequences):
        length = len(sequences[0])
        self.calls.append((len(sequences), length))
        values = torch.tensor(sequences, dtype=torch.float32)
        hidden = torch.stack((values, values + 100), dim=-1)
        return LayerBatch(
            hidden=hidden,
            mask=torch.ones(values.shape, dtype=torch.long),
        )


class _Writer:
    def __init__(self) -> None:
        self.rows = []
        self.hidden = []
        self.means = []
        self.orders = []
        self.waited = None

    def add_batch(self, rows, hidden, means, order_indices=None):
        self.rows.extend(rows)
        self.hidden.append(hidden.clone())
        self.means.append(means.clone())
        self.orders.extend(order_indices)

    def wait_until_order(self, expected):
        self.waited = expected


def test_length_bucket_forward_preserves_pair_major_writer_order() -> None:
    rows = [
        row(0, (1, 2), (3, 4, 5)),
        row(1, (6,), (7, 8)),
        row(2, (9, 10), (11,)),
    ]
    extractor = _Extractor()
    writer = _Writer()
    stats = process_window_by_length(
        extractor,
        writer,
        [(value, index) for index, value in enumerate(rows)],
        schedule=ForwardSchedule(
            max_batch_size=16,
            token_budget=32,
            writer_batch_tokens=8,
        ),
    )
    assert writer.rows == rows
    assert writer.orders == [0, 1, 2]
    assert writer.waited == 3
    assert stats.model_calls == 3
    assert sorted(extractor.calls) == [(1, 3), (2, 1), (3, 2)]
    packed = torch.cat(writer.hidden)
    expected_tokens = []
    for value in rows:
        expected_tokens.extend(value.input_ids_a)
        expected_tokens.extend(value.input_ids_b)
    assert packed[:, 0].tolist() == expected_tokens


def test_sparse_decoder_matches_dense_values_and_gradients() -> None:
    dense = BatchTopKSAE(4, 8, 2, decoder_backend="dense")
    sparse = BatchTopKSAE(4, 8, 2, decoder_backend="sparse")
    sparse.load_state_dict(copy.deepcopy(dense.state_dict()))
    features_dense = torch.zeros(5, 8, requires_grad=True)
    features_dense.data[:, :2] = torch.randn(5, 2)
    features_sparse = features_dense.detach().clone().requires_grad_(True)
    left = dense.decode(features_dense)
    right = sparse.decode(features_sparse)
    torch.testing.assert_close(left, right)
    left.square().sum().backward()
    right.square().sum().backward()
    active = features_dense.detach() != 0
    torch.testing.assert_close(
        features_dense.grad[active],
        features_sparse.grad[active],
    )
    assert bool((features_sparse.grad[~active] == 0).all())
    torch.testing.assert_close(
        dense.decoder_weight.grad,
        sparse.decoder_weight.grad,
    )


def test_chunk_input_dedup_expands_to_exact_same_preactivations() -> None:
    model = BatchTopKSAE(3, 6, 2)
    inputs = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0],
        ]
    )
    batch = {
        "pair_id": torch.tensor([10, 10, 11, 11]),
        "side": torch.tensor([False, False, True, True]),
    }
    unique_rows, inverse = chunk_input_deduplication(batch, len(inputs))
    direct = model.pre_activations(inputs)
    deduplicated = model.pre_activations(
        inputs.index_select(0, unique_rows)
    ).index_select(0, inverse)
    assert torch.equal(direct, deduplicated)


def test_feature_tensor_parallel_single_rank_checkpoint(tmp_path) -> None:
    model = FeatureTensorParallelSAE(4, 8, 2)
    model.initialize_from_global_seed(123)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(3, 4)
    output = model(inputs)[0]
    output.square().mean().backward()
    optimizer.step()
    checkpoint = tmp_path / "tp"
    save_sharded_training_checkpoint(
        model=model,
        optimizer=optimizer,
        output_dir=checkpoint,
        state={
            "step": 1,
            "samples_seen": 3,
            "config": {
                "sae_training_implementation_version": (
                    SAE_TRAINING_IMPLEMENTATION_VERSION
                ),
                "sae_parameter_schema_version": SAE_PARAMETER_SCHEMA_VERSION,
            },
        },
    )
    restored = FeatureTensorParallelSAE(4, 8, 2)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    state = load_sharded_training_checkpoint(
        model=restored,
        optimizer=restored_optimizer,
        input_dir=checkpoint,
    )
    assert state["step"] == 1
    for left, right in zip(
        model.local.state_dict().values(),
        restored.local.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_auxk_bfloat16_cuda_backward_is_finite() -> None:
    model = BatchTopKSAE(32, 128, 4, decoder_backend="sparse").cuda()
    model.num_occurrences_since_fired.fill_(1)
    inputs = torch.randn(32, 32, device="cuda")
    targets = torch.randn_like(inputs)
    candidate_ids = model.auxiliary_feature_ids(1, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reconstructed, _, _, _, _, auxiliary = model(
            inputs,
            batch_topk=True,
            return_activity_counts=True,
            auxiliary_feature_ids=candidate_ids,
            auxiliary_k=8,
            return_auxiliary=True,
        )
        primary = (reconstructed.float() - targets).square().mean()
        auxiliary_loss = (
            auxiliary.float() - (targets - reconstructed.detach())
        ).square().mean()
        loss = primary + 0.0625 * auxiliary_loss
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert model.encoder_weight.grad is not None
    assert model.decoder_weight.grad is not None
    assert bool((model.encoder_weight.grad[candidate_ids].abs().sum() > 0))
    assert bool((model.decoder_weight.grad[:, candidate_ids].abs().sum() > 0))
