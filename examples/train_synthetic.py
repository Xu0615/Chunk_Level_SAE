#!/usr/bin/env python3
"""Small CPU demonstration of Mean, Cross, and nested Joint training.

Synthetic paired chunk means share latent factors and have independent noise.
This checks the SAE APIs and checkpoint round-trip, not language-model quality.
The production trainer adds activation-cache provenance, AuxK, exact occurrence
coverage, distributed optimization, and held-out checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from chunk_saes.sae import BatchTopKSAE, JointChunkSAE, load_sae, save_sae


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("choose an empty output directory to preserve earlier runs")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    hidden, width, k, prefix, alpha = 32, 128, 8, 64, 0.25
    ground_truth = F.normalize(torch.randn(24, hidden), dim=1)

    def pair(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.rand(count, 24)
        latent = latent * (torch.rand_like(latent) < 0.15)
        shared = latent @ ground_truth
        return shared + 0.1 * torch.randn_like(shared), shared + 0.1 * torch.randn_like(shared)

    a, b = pair(2048)
    x, partner = torch.cat((a, b)), torch.cat((b, a))
    test_a, test_b = pair(256)
    test_x, test_partner = torch.cat((test_a, test_b)), torch.cat((test_b, test_a))
    mean_baseline = (x - x.mean(0)).square().mean().clamp_min(1e-8)
    cross_baseline = (partner - partner.mean(0)).square().mean().clamp_min(1e-8)
    results = {}
    for mode in ("mean", "cross", "joint_chunk"):
        torch.manual_seed(args.seed)
        model = (JointChunkSAE(hidden, width, k, cross_prefix=prefix)
                 if mode == "joint_chunk" else BatchTopKSAE(hidden, width, k))
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        with torch.no_grad():
            model.pre_bias.copy_(x.mean(0))
            model.decoder_bias.copy_((partner if mode == "cross" else x).mean(0))
            if isinstance(model, JointChunkSAE):
                model.decoder_cross_bias.copy_(partner.mean(0))
        model.train()
        for step in range(args.steps):
            ids = torch.randint(len(x), (128,))
            if isinstance(model, JointChunkSAE):
                own, other, features, threshold, *_ = model.forward_joint(x[ids], batch_topk=True)
                loss = (F.mse_loss(own, x[ids]) / mean_baseline
                        + alpha * F.mse_loss(other, partner[ids]) / cross_baseline) / (1 + alpha)
            else:
                reconstructed, features, threshold = model(x[ids], batch_topk=True)
                target = partner[ids] if mode == "cross" else x[ids]
                loss = F.mse_loss(reconstructed, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{mode} produced non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.remove_parallel_decoder_gradient_()
            optimizer.step()
            model.normalize_decoder_()
            with torch.no_grad():
                if step == 0:
                    model.threshold.copy_(threshold)
                else:
                    model.threshold.lerp_(threshold, 0.1)
        model.eval()
        with torch.inference_mode():
            z = model.encode(test_x)
            target = test_partner if mode == "cross" else test_x
            report = {"heldout_mse": float(F.mse_loss(model.decode(z), target)),
                      "mean_active_features": float((z != 0).sum(1).float().mean())}
            if isinstance(model, JointChunkSAE):
                report["heldout_cross_mse"] = float(F.mse_loss(model.decode_cross(z), test_partner))
        config = {"activation_dim": hidden, "dict_size": width, "k": k,
                  "mode": mode, "decoder_backend": "dense", "synthetic_demo": True}
        if mode == "joint_chunk":
            config.update(joint_chunk_layout="nested_prefix", joint_cross_prefix=prefix,
                          joint_chunk_alpha=alpha)
        destination = args.output_dir / mode
        save_sae(model, destination, config)
        reloaded = load_sae(destination).model
        with torch.inference_mode():
            torch.testing.assert_close(reloaded.encode(test_x), z)
        results[mode] = report
        print(f"{mode:12s} {json.dumps(report, sort_keys=True)}")
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved three reloadable synthetic checkpoints to {args.output_dir}")


if __name__ == "__main__":
    main()
