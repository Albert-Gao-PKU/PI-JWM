import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pi_jwm.step4_2c_c_flow_sample_tensor_v1 import load_flow_tensor_batch
from pi_jwm.step4_3a_typed_dual_graph_builder_v1 import load_typed_dual_graph_batch
from pi_jwm.step4_3b_dual_graph_encoder_v1 import PIJointGraphEncoder, DualGraphEncoderConfig, fit_encoder_normalization_stats
from pi_jwm.research_geo_temporal_count_v1 import (
    ResearchEncoderConfig, GeoTemporalCountEncoder, save_research_encoder, load_research_encoder,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "code/artifacts/protocols"


class ResearchEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tensor = load_flow_tensor_batch(PROTOCOL / "pi_jwm_step4_2c_c_flow_sample_tensor_v1_20260920/flow_tensor.npz")
        cls.graph = load_typed_dual_graph_batch(PROTOCOL / "pi_jwm_step4_3a_typed_dual_graph_builder_v1_20260920/typed_dual_graph_batch.npz")
        cls.stats = fit_encoder_normalization_stats(cls.tensor, cls.graph, PROTOCOL / "pi_jwm_step4_2a_graph_input_extension_v1_20260920/train_normalization_stats.json")

    def model(self, **options):
        return GeoTemporalCountEncoder(ResearchEncoderConfig(**options), self.tensor["contract"], self.graph["contract"], self.stats)

    def test_all_switches_off_match_baseline_bitwise(self):
        base = PIJointGraphEncoder(DualGraphEncoderConfig(), self.tensor["contract"], self.graph["contract"], self.stats)
        candidate = self.model(enable_geometry_branch=False, enable_csi_history=False, enable_count_aggregation=False)
        self.assertEqual(set(base.state_dict()), set(candidate.state_dict()))
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]), name)
        a, b = base(self.tensor, self.graph), candidate(self.tensor, self.graph)
        for section in ("physical", "information", "structural", "diagnostics"):
            for name in a[section]:
                if isinstance(a[section][name], dict):
                    for key in a[section][name]:
                        self.assertTrue(torch.equal(a[section][name][key], b[section][name][key]))
                else:
                    self.assertTrue(torch.equal(a[section][name], b[section][name]), name)

    def test_geometry_translation_invariant_and_masked(self):
        model = self.model()
        a, mask = model.geometry_features(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        changed["entity_position_raw"] += 1000
        b, second_mask = model.geometry_features(changed, self.graph)
        self.assertTrue(torch.equal(mask, second_mask))
        self.assertTrue(torch.allclose(a, b, atol=2e-4))
        changed["entity_position_mask"][:] = False
        values, masks = model.geometry_features(changed, self.graph)
        self.assertFalse(bool(masks[..., :4].any()))
        self.assertTrue(bool((values[..., :4] == 0).all()))

    def test_geometry_no_matching_physical_edge_needed(self):
        model = self.model()
        graph = copy.deepcopy(self.graph)
        graph["blocks"]["physical_relations"]["presence"][:] = False
        raw, mask = model.geometry_features(self.tensor, graph)
        self.assertGreater(int(mask.sum()), 0)

    def test_history_interface_and_sensitivity(self):
        model = self.model(enable_geometry_branch=False, enable_count_aggregation=False)
        out = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        active = changed["comm_csi_mask"][:, 0] & changed["comm_csi_mask"][:, -1]
        self.assertTrue(bool(active.any()))
        changed["comm_csi_raw"][:, 0][active] += 20
        second = model(changed, self.graph)
        self.assertFalse(torch.equal(out["information"]["comm_relation_latent"], second["information"]["comm_relation_latent"]))

    def test_current_endpoint_mismatch_and_rb_misorder_rejected(self):
        model = self.model()
        active = self.tensor["comm_relation_presence"][:, 0] & self.tensor["comm_relation_presence"][:, -1]
        bi, ri = np.argwhere(active)[0]
        changed = copy.deepcopy(self.tensor)
        changed["comm_source_index"][bi, -1, ri] += 1
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            model(changed, self.graph)
        changed = copy.deepcopy(self.tensor)
        changed["comm_rb_indices"][changed["comm_rb_mask"]] += 1
        with self.assertRaisesRegex(ValueError, "RB ordering"):
            model(changed, self.graph)

    def test_historical_row_permutation_does_not_mix_links(self):
        model = self.model()
        expected = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        width = changed["comm_csi_raw"].shape[2]
        permutation = np.arange(width)[::-1]
        for key, value in changed.items():
            if key.startswith("comm_") and isinstance(value, np.ndarray) and value.ndim >= 3 and value.shape[2] == width:
                value[:, 0] = value[:, 0, permutation].copy()
        actual = model(changed, self.graph)
        self.assertTrue(torch.equal(expected["information"]["comm_relation_latent"], actual["information"]["comm_relation_latent"]))

    def test_duplicate_history_link_is_rejected(self):
        model = self.model()
        changed = copy.deepcopy(self.tensor)
        active = changed["comm_relation_presence"][0, 0] & changed["comm_relation_validity"][0, 0]
        first, second = np.flatnonzero(active)[:2]
        for key in ("comm_source_index", "comm_target_index", "comm_relation_type_index"):
            changed[key][0, 0, second] = changed[key][0, 0, first]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            model(changed, self.graph)

    def test_new_model_entity_task_flow_permutation_equivariance(self):
        from test_step4_3b_dual_graph_encoder_v1 import Step43BDualGraphEncoderTests
        Step43BDualGraphEncoderTests.test_entity_index_permutation_is_equivariant(self)

    def test_masked_csi_placeholders_do_not_affect_latents(self):
        model = self.model()
        expected = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        changed["comm_csi_raw"][~changed["comm_csi_mask"]] = np.nan
        actual = model(changed, self.graph)
        self.assertTrue(torch.equal(expected["information"]["comm_relation_latent"], actual["information"]["comm_relation_latent"]))

    def test_ablation_switches_preserve_output_shapes(self):
        from run_research_geo_temporal_count_v1 import VARIANTS
        for variant, flags in VARIANTS.items():
            model = self.model(enable_geometry_branch=flags[0], enable_csi_history=flags[1], enable_count_aggregation=flags[2])
            output = model(self.tensor, self.graph)
            self.assertEqual(output["information"]["comm_relation_latent"].shape[-1], 16, variant)
            self.assertTrue(bool(torch.isfinite(output["information"]["agent_latent"]).all()), variant)

    def test_count_distinguishes_duplicate_messages_and_zero_neighbours(self):
        model = self.model()
        messages = torch.ones(1, 2, model.config.d_h)
        one, _ = model._aggregate(messages, torch.zeros(1, 2, dtype=torch.long), torch.tensor([[True, False]]), 2, "flow", "forward")
        two, _ = model._aggregate(messages, torch.zeros(1, 2, dtype=torch.long), torch.tensor([[True, True]]), 2, "flow", "forward")
        self.assertFalse(torch.equal(one[:, 0], two[:, 0]))
        self.assertTrue(torch.equal(two[:, 1], torch.zeros_like(two[:, 1])))
        reversed_messages, _ = model._aggregate(messages.flip(1), torch.zeros(1, 2, dtype=torch.long), torch.tensor([[True, True]]), 2, "flow", "forward")
        self.assertTrue(torch.equal(two, reversed_messages))

    def test_future_targets_ignored_and_wired_p2c_zero(self):
        model = self.model()
        a = model(self.tensor, self.graph)
        changed = copy.deepcopy(self.tensor)
        changed["target_future_csi"] = np.full((2, 3), 999999)
        b = model(changed, self.graph)
        self.assertTrue(torch.equal(a["information"]["comm_relation_latent"], b["information"]["comm_relation_latent"]))
        wired = self.graph["blocks"]["comm_relations"]["relation_type_index"] != 2
        self.assertTrue(bool((a["diagnostics"]["p2c_message"][wired] == 0).all()))

    def test_finite_gradients_all_new_branches_and_roundtrip(self):
        model = self.model()
        output = model(self.tensor, self.graph)
        loss = sum(value[..., 0].sum() for value in output["information"].values())
        loss.backward()
        for prefix in ("geometry_encoder", "comm_history_gru", "comm_history_fuse", "count_fuse"):
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertTrue(grads, prefix)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads), prefix)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, prefix)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "encoder.pt"
            save_research_encoder(model, path)
            restored = load_research_encoder(path)
            reloaded = restored(self.tensor, self.graph)
            for key in output["information"]:
                self.assertTrue(torch.equal(output["information"][key], reloaded["information"][key]))


if __name__ == "__main__":
    unittest.main()
