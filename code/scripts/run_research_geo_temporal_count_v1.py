"""Separate research entry point; default is bounded readiness, not formal training.

Run with PYTHONPATH=code/src:code/scripts. Every ablation uses the same original
FullFormalTrainer sampler, RSSM, losses, targets and formal optimization schedule.
Readiness uses a deliberately compressed schedule and MUST NOT be compared as
a trained performance result. mode=train requires an explicit CLI choice.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from pi_jwm.research_geo_temporal_count_v1 import ResearchEncoderConfig, ResearchFullFormalTrainer
from pi_jwm.step5_2_training_loop_v1 import Step52TrainingConfig, CurriculumConfig, KLSchedule
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from pi_jwm.step5_6b_formal_runner_v1 import validate_full

ROOT = Path(__file__).resolve().parents[2]
BASELINE_SOURCE_COMMIT = "f95bf2e5b9430666f325af3e6d6846d68b97e6e8"
DEFAULT_MANIFEST = ROOT / "code/artifacts/formal_dataset/pi_jwm_formal_dataset_v1_h2_l4_20260923_causalfix1/formal_dataset_manifest.json"
VARIANTS = {
    "baseline": (False, False, False), "geometry": (True, False, False),
    "history": (False, True, False), "count": (False, False, True),
    "geometry_history": (True, True, False), "full": (True, True, True),
}


def deployment_git_commit():
    """Record a real Git HEAD when available; extracted archives have no HEAD.

    The baseline source commit is recorded separately, never substituted for
    the uncommitted research code. Exact deployed source SHA256 values remain
    authoritative for identifying that code.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def gradient_audit(trainer):
    audit = {}
    for prefix in ("geometry_encoder", "p2c_gate", "p2c_value", "comm_history_gru", "comm_history_fuse", "count_fuse"):
        params = [p for name, p in trainer.encoder.named_parameters() if name.startswith(prefix)]
        grads = [p.grad for p in params if p.grad is not None]
        audit[prefix] = {"enabled": bool(params), "gradient_present": bool(grads),
                         "finite": all(bool(torch.isfinite(g).all()) for g in grads),
                         "abs_sum": sum(float(g.detach().abs().sum()) for g in grads)}
    return audit


def optimize_batch(trainer, indices, horizon):
    trainer.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    result = trainer._run_batch(indices, horizon, stage="prior_dominant_recursive", beta_kl=0.0, training=True)
    loss = result["L_Total"]
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite research loss")
    loss.backward()
    if not trainer._finite_gradients(trainer.parameters()):
        raise FloatingPointError("nonfinite research gradient")
    audit = gradient_audit(trainer)
    before = trainer.parameter_digest()
    torch.nn.utils.clip_grad_norm_(trainer.parameters(), trainer.config.gradient_clip_norm)
    trainer.optimizer.step()
    return {"horizon": horizon, "L_Total": float(loss.detach()), "L_Pred": float(result["L_Pred"].detach()),
            "loss_finite": True, "gradients_finite": True, "parameter_changed": before != trainer.parameter_digest(),
            "new_branch_gradients": audit, "prior_only_rollout": bool(result["rollout"]["prior_only"]),
            "future_target_consumed_by_rollout": bool(result["rollout"]["future_target_consumed_by_rollout"])}


def readiness(trainer, args, provenance):
    receipt = {"status": "running", "provenance": provenance, "scope": "training_readiness_not_performance", "checks": {}}
    path = args.output / "readiness.json"
    write(path, receipt)
    # All 60 tensor/graph shards: history/current identity and full payload hashes.
    count, wireless_history_rows = 0, 0
    reused = None
    if args.reuse_interface_audit:
        reused = json.loads(args.reuse_interface_audit.read_text(encoding="utf-8"))
        prior = reused["provenance"]
        previous_audit = reused.get("checks", {}).get("all_shards", {})
        encoder_source = "code/src/pi_jwm/research_geo_temporal_count_v1.py"
        if (prior["dataset"]["dataset_manifest_hash"] != provenance["dataset"]["dataset_manifest_hash"]
            or prior["dataset"]["encoder_normalization_sha256"] != provenance["dataset"]["encoder_normalization_sha256"]
            or prior["dataset"]["research_source_sha256"][encoder_source] != provenance["dataset"]["research_source_sha256"][encoder_source]
            or prior["config"]["encoder"]["enable_csi_history"] != provenance["config"]["encoder"]["enable_csi_history"]
            or previous_audit.get("windows") != 5520 or previous_audit.get("payload_hashes_verified") != 60):
            raise ValueError("cached interface audit provenance mismatch")
        # Rehash every payload even when reusing identity/alignment inspection.
        # A source receipt that failed later training can still contain a valid
        # earlier all-shard audit; it never substitutes for a training gate.
        from pi_jwm.step5_2_training_loop_v1 import _sha
        for tid, row in trainer.shards.shards.items():
            for family in ("samples", "tensor", "graph", "target"):
                entry = row["files"][family]
                if _sha(trainer.shards.paths[family] / entry["path"]) != entry["sha256"]:
                    raise ValueError(f"cached interface audit payload changed: {tid}/{family}")
            trainer.shards._verified_shards.add(tid)
        count, wireless_history_rows = previous_audit["windows"], previous_audit["wireless_history_rows"]
    for tid, row in (() if reused else trainer.shards.shards.items()):
        _, tensor, graph, _ = trainer.shards._load_shard(tid)
        # CPU audit avoids uploading all 92 windows to limited laptop VRAM.
        encoder = trainer.encoder
        if encoder.config.enable_csi_history:
            device = next(encoder.parameters()).device
            encoder.cpu()
            try:
                _, mask, active = encoder.audit_comm_history(tensor, graph["blocks"]["comm_relations"])
                wireless_history_rows += int((active & mask.any(-1)).sum())
            finally:
                encoder.to(device)
        count += row["sample_count"]
        if len(trainer.shards._verified_shards) % 10 == 0:
            print(json.dumps({"audited_shards": len(trainer.shards._verified_shards)}), flush=True)
    receipt["checks"]["all_shards"] = {"shards": len(trainer.shards.shards), "windows": count,
        "payload_hashes_verified": len(trainer.shards._verified_shards), "wireless_history_rows": wireless_history_rows,
        "source_audit_reused": str(args.reuse_interface_audit) if reused else None, "all_payloads_rehashed": True}
    print(json.dumps({"all_shards_audited": count}), flush=True)
    write(path, receipt)

    steps = []
    for step in range(args.steps):
        began = time.perf_counter()
        row = trainer.train_step(global_step=step, epoch=0)
        brief = {key: row[key] for key in ("global_step", "stage", "rollout_horizon", "L_Total", "L_Pred", "loss_finite", "gradients_finite", "parameter_update", "prior_only_rollout", "future_state_not_rollout_input")}
        brief["wall_seconds"] = time.perf_counter() - began
        brief["new_branch_gradients"] = gradient_audit(trainer)
        steps.append(brief)
        receipt["short_training"] = steps
        write(path, receipt)
        print(json.dumps({"short_step": brief}), flush=True)

    cross = [trainer.data.train_indices[0], trainer.data.train_indices[92]]
    receipt["cross_trajectory_h4"] = optimize_batch(trainer, cross, 4)
    receipt["cross_trajectory_h4"]["trajectory_ids"] = [trainer.shards.index[i]["metadata"]["trajectory_id"] for i in cross]
    print(json.dumps({"cross_h4": receipt["cross_trajectory_h4"]}), flush=True)

    # Fixed small TRAIN batch and deterministic seed: learnability only.
    fixed = trainer.formal_sampler.batch_for_global_step(0, args.batch_size)
    losses = []
    for index in range(args.overfit_steps + 1):
        torch.manual_seed(10001)
        if trainer.device.type == "cuda":
            torch.cuda.manual_seed_all(10001)
        trainer.eval()
        with torch.no_grad():
            result = trainer._run_batch(fixed, 1, stage="prior_dominant_recursive", beta_kl=0.0, training=False)
            losses.append(float(result["L_Pred"]))
        if index < args.overfit_steps:
            optimize_batch(trainer, fixed, 1)
    receipt["tiny_overfit"] = {"steps": args.overfit_steps, "horizon": 1, "prediction_loss": losses,
                               "prediction_loss_reduced": losses[-1] < losses[0], "formal_generalization_claim": False}
    print(json.dumps({"tiny_overfit": receipt["tiny_overfit"]}), flush=True)

    # Probe the actual baseline batch size, even when bounded short training
    # uses a smaller batch. No architecture or precision reduction is applied.
    began = time.perf_counter()
    receipt["formal_batch8_h4"] = optimize_batch(trainer, trainer.formal_sampler.batch_for_global_step(0, 8), 4)
    receipt["formal_batch8_h4"]["batch_size"] = 8
    receipt["formal_batch8_h4"]["wall_seconds"] = time.perf_counter() - began
    print(json.dumps({"batch8_h4": receipt["formal_batch8_h4"]}), flush=True)

    digest = trainer.parameter_digest()
    validation = trainer.validate_indices(trainer.data.validation_indices[:2])
    receipt["validation_interface"] = {"sample_ids": validation["sample_ids"], "prior_only": validation["prior_only"],
        "horizon_rows": validation["horizon_rows"], "parameters_unchanged": digest == trainer.parameter_digest(),
        "full_1104_window_validation": False}
    state = trainer.state_snapshot(global_step=args.steps, epoch=0, best_l_val=1e30)
    checkpoint = args.output / "readiness_checkpoint.pt"
    trainer.save_checkpoint(checkpoint, state=state)
    # Fresh instance verifies factory AND optimizer binding, not just in-place reload.
    restored = ResearchFullFormalTrainer(trainer.config, trainer.shards.load_batch([fixed[0]]), encoder_stats=trainer.encoder.normalization_stats)
    restored.shards = trainer.shards
    restored.formal_sampler = trainer.formal_sampler
    restored.data.samples = list(trainer.shards.index)
    restored.data.train_indices = list(trainer.data.train_indices)
    restored.data.validation_indices = list(trainer.data.validation_indices)
    restored.data.identity = trainer.data.identity
    restored_state = restored.load_checkpoint(checkpoint)
    receipt["fresh_checkpoint_reload_identical"] = trainer.parameter_digest() == restored.parameter_digest() and restored_state == state
    before = {name: value.detach().clone() for name, value in restored.encoder.named_parameters()}
    restored.train_step(global_step=0, epoch=0)
    receipt["resumed_optimizer_updates_encoder"] = any(not torch.equal(before[n], p) for n, p in restored.encoder.named_parameters())
    gradient_rows = receipt["cross_trajectory_h4"]["new_branch_gradients"]
    receipt["passed"] = bool(count == 5520 and len(trainer.shards._verified_shards) == 60
        and all(r["loss_finite"] and r["gradients_finite"] and r["parameter_update"]["any_changed"] for r in steps)
        and {1, 2, 4}.issubset({r["rollout_horizon"] for r in steps})
        and all(not r["enabled"] or (r["finite"] and r["gradient_present"] and r["abs_sum"] > 0) for r in gradient_rows.values())
        and receipt["cross_trajectory_h4"]["parameter_changed"]
        and receipt["formal_batch8_h4"]["parameter_changed"]
        and receipt["tiny_overfit"]["prediction_loss_reduced"]
        and validation["prior_only"] and receipt["validation_interface"]["parameters_unchanged"]
        and receipt["fresh_checkpoint_reload_identical"] and receipt["resumed_optimizer_updates_encoder"])
    if trainer.device.type == "cuda":
        receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    receipt["status"] = "passed" if receipt["passed"] else "failed"
    write(path, receipt)
    if not receipt["passed"]:
        raise RuntimeError(f"readiness gate failed; inspect {path}")


def train(trainer, args, provenance):
    """Opt-in formal research run, never called by readiness mode."""
    state = trainer.load_checkpoint(args.resume) if args.resume else trainer.state_snapshot(best_l_val=1e30)
    start = int(state["global_step"])
    with (args.output / "training.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start, args.steps):
            result = trainer.train_step(global_step=step, epoch=step // 552)
            log.write(json.dumps(result, allow_nan=False) + "\n")
            log.flush()
            state = trainer.state_snapshot(global_step=step + 1, epoch=step // 552, best_l_val=state["best_l_val"],
                early_stopping_counter=state.get("early_stopping_counter", 0))
            if (step + 1) % 552 == 0 or step + 1 == args.steps:
                trainer.save_checkpoint(args.output / "latest.pt", state=state)
            if (step + 1) % 1104 == 0 or step + 1 == args.steps:
                validation = validate_full(trainer, lambda done: print(json.dumps({"validation_windows": done}), flush=True))
                write(args.output / f"validation_{step+1}.json", validation)
                previous_best = state["best_l_val"]
                state = trainer.update_validation_state(state, validation)
                if validation["L_Val"] < previous_best:
                    trainer.save_checkpoint(args.output / "best.pt", state=state)
                trainer.save_checkpoint(args.output / "latest.pt", state=state)
                if state["early_stopping_should_stop"]:
                    write(args.output / "complete.json", {"status": "early_stopped", "provenance": provenance,
                        "global_step": step + 1, "best_L_Val": state["best_l_val"]})
                    return
            print(json.dumps({"step": step + 1, "loss": result["L_Total"]}), flush=True)
    write(args.output / "complete.json", {"provenance": provenance, "global_step": args.steps, "best_L_Val": state["best_l_val"]})


def make_config(args):
    flags = VARIANTS[args.variant]
    encoder = ResearchEncoderConfig(enable_geometry_branch=flags[0], enable_csi_history=flags[1], enable_count_aggregation=flags[2])
    return Step52TrainingConfig(seed=5601, learning_rate=3e-4, batch_size=args.batch_size,
        max_steps=args.steps, max_epochs=10 if args.mode == "train" else 1, max_horizon=4,
        stage1_steps=552 if args.mode == "train" else 2,
        curriculum=CurriculumConfig(horizons=(1, 2, 4), start_steps=(0, 1104, 2208) if args.mode == "train" else (0, 2, 4)),
        kl_schedule=KLSchedule(target_beta=1.0, warmup_steps=1104 if args.mode == "train" else 8, free_bits=0.1),
        encoder=encoder, device=args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("readiness", "train"), default="readiness")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--overfit-steps", type=int, default=12)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reuse-interface-audit", type=Path,
                        help="reuse matching all-shard identity audit, after rehashing every payload")
    args = parser.parse_args()
    args.steps = args.steps or (8 if args.mode == "readiness" else 5520)
    if args.steps <= 0 or args.batch_size <= 0 or args.overfit_steps < 1:
        parser.error("steps, batch size and overfit steps must be positive")
    if args.mode == "readiness" and (args.steps < 5 or args.resume):
        parser.error("readiness needs >=5 steps and starts from fresh weights")
    if args.mode == "train" and args.device != "cuda":
        parser.error("formal research training requires CUDA; use readiness for CPU checks")
    if args.mode == "train" and args.batch_size != 8:
        parser.error("formal comparison preserves baseline batch size 8")
    # Avoid silently overwriting evidence or a previous experiment.
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("output is not empty; choose a new directory or explicitly resume")
    torch.set_num_threads(2)
    config = make_config(args)
    print(json.dumps({"initializing": True, "mode": args.mode, "variant": args.variant, "device": args.device}), flush=True)
    interface = FormalTrainingInterface.from_manifest(args.manifest)
    trainer = ResearchFullFormalTrainer.from_interface(interface, config)
    source_paths = [Path(__file__), ROOT / "code/src/pi_jwm/research_geo_temporal_count_v1.py",
        ROOT / "code/src/pi_jwm/step4_3b_dual_graph_encoder_v1.py", ROOT / "code/src/pi_jwm/step5_2_training_loop_v1.py",
        ROOT / "code/src/pi_jwm/step5_5_full_sharded_loader_v1.py", ROOT / "code/src/pi_jwm/step4_4_structured_rssm_world_model_v1.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    trainer.data.identity["research_source_sha256"] = hashes
    provenance = {"variant": args.variant, "config": asdict(config), "dataset": trainer.data.identity,
        "git_commit": deployment_git_commit(), "baseline_source_commit": BASELINE_SOURCE_COMMIT,
        "torch": torch.__version__, "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "model_parameters": sum(p.numel() for p in trainer.parameters()),
        "encoder_parameters": sum(p.numel() for p in trainer.encoder.parameters()),
        "formal_performance_claim": False}
    write(args.output / "provenance.json", provenance)
    try:
        (readiness if args.mode == "readiness" else train)(trainer, args, provenance)
    except Exception as exc:
        write(args.output / "failure.json", {"error_type": type(exc).__name__, "message": str(exc), "provenance": provenance})
        raise


if __name__ == "__main__":
    main()
