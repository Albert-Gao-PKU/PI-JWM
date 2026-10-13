"""Focused regression receipt and optional real-checkpoint action sensitivity."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

import torch

from pi_jwm.research_geo_temporal_count_v1 import ResearchEncoderConfig, ResearchFullFormalTrainer
from pi_jwm.step4_4_structured_rssm_world_model_v1 import StructuredRSSMConfig
from pi_jwm.step5_2_training_loop_v1 import Step52TrainingConfig, CurriculumConfig, KLSchedule
from pi_jwm.step5_4_formal_training_readiness_v1 import FormalTrainingInterface
from run_research_geo_temporal_count_v1 import DEFAULT_MANIFEST, ROOT, write


def action_sensitivity(path):
    package = torch.load(path, map_location="cpu", weights_only=False)
    values = package["config"]
    curriculum = values["curriculum"]
    config = Step52TrainingConfig(**{**values,
        "encoder": ResearchEncoderConfig(**values["encoder"]),
        "rssm": StructuredRSSMConfig(**values["rssm"]),
        "curriculum": CurriculumConfig(tuple(curriculum["horizons"]), tuple(curriculum["start_steps"])),
        "kl_schedule": KLSchedule(**values["kl_schedule"])})
    # Check executable source identity before treating the checkpoint as current.
    hashes = package["data_identity"]["research_source_sha256"]
    for relative, digest in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"checkpoint executable source changed: {relative}")
    interface = FormalTrainingInterface.from_manifest(DEFAULT_MANIFEST)
    trainer = ResearchFullFormalTrainer.from_interface(interface, config)
    trainer.data.identity["research_source_sha256"] = hashes
    trainer.load_checkpoint(path)
    ids = trainer.formal_sampler.batch_for_global_step(0, 1)
    trainer.data = trainer.shards.load_batch(ids)
    trainer._move_data_to_device()
    trainer.eval()
    from build_step5_1d_unified_model_chain_v1 import build_action
    with torch.no_grad():
        state, graph = trainer._batch_initial_state([0])
        zpi = trainer.encoder(trainer.data.tensor, trainer.data.graph)
        latent = trainer.model.initialize_latent(zpi, state, posterior_mode="mean")
        action, _ = build_action(trainer.data.samples[0], state, 0)
        active_uav = torch.nonzero(state["uav_mask"][0] & state["entity_presence"][0], as_tuple=False).flatten()
        if active_uav.numel() == 0:
            raise ValueError("real action sensitivity anchor has no UAV")
        a, b = {k: v.clone() for k, v in action.items()}, {k: v.clone() for k, v in action.items()}
        a["mobility_entity_index"] = active_uav[None]
        b["mobility_entity_index"] = active_uav[None]
        a["mobility_values"] = state["position"].new_tensor([0.0, 0.0, 4.0, 1.0]).view(1, 1, 4).expand(1, len(active_uav), 4).clone()
        b["mobility_values"] = a["mobility_values"].clone()
        b["mobility_values"][..., 0] = torch.pi / 2
        # Two explicit controls on real current state, not a planner evaluation
        # or a claim that these particular joint actions were observed in TRAIN.
        la, sa, ga, ta = trainer.model.one_step(latent, state, graph, a, prior_mode="mean", service_mode="expectation", generator=None)
        lb, sb, gb, tb = trainer.model.one_step(latent, state, graph, b, prior_mode="mean", service_mode="expectation", generator=None)
        state_changed = not torch.equal(sa["position"], sb["position"])
        recurrent_changed = not torch.equal(la["h"]["physical"], lb["h"]["physical"])
        finite = all(bool(torch.isfinite(x).all()) for x in (sa["position"], sb["position"], la["h"]["physical"], lb["h"]["physical"]))
    return {"passed": state_changed and recurrent_changed and finite, "current_state_source": "real TRAIN window",
        "sample_id": trainer.data.samples[0]["metadata"]["sample_id"], "checkpoint_reconstructed_from_fresh_factory": True,
        "uav_count": len(active_uav), "position_action_sensitive": state_changed, "recurrent_state_action_sensitive": recurrent_changed,
        "outputs_finite": finite, "future_target_used": False, "planner_or_performance_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    tests = ROOT / "code/tests"
    sys.path.insert(0, str(tests))
    patterns = ["test_research_geo_temporal_count_v1.py", "test_research_training_entry_v1.py",
                "test_step4_3b_dual_graph_encoder_v1.py", "test_step4_4_structured_rssm_world_model_v1.py",
                "test_step5_6b_formal_runner_v1.py"]
    suites = []
    for pattern in patterns:
        suite = unittest.defaultTestLoader.discover(str(tests), pattern=pattern)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        suites.append({"pattern": pattern, "tests_run": result.testsRun, "passed": result.wasSuccessful(),
                       "failures": [trace for _, trace in result.failures], "errors": [trace for _, trace in result.errors]})
    receipt = {"scope": "focused_not_whole_repository", "suites": suites,
        "tests_run": sum(row["tests_run"] for row in suites), "passed": all(row["passed"] for row in suites),
        "historical_suite_gap": "Step5.2 tests cannot initialize: historical unified_flow_tensor.npz is absent; not counted as passing"}
    if args.checkpoint:
        receipt["action_sensitivity"] = action_sensitivity(args.checkpoint)
        receipt["passed"] &= receipt["action_sensitivity"]["passed"]
    write(args.output, receipt)
    print(json.dumps({"passed": receipt["passed"], "tests_run": receipt["tests_run"], "output": str(args.output)}))
    raise SystemExit(0 if receipt["passed"] else 1)


if __name__ == "__main__":
    main()
