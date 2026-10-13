"""Opt-in research encoder. Existing graph/RSSM/target interfaces are unchanged.

These modules are hypotheses to test, not demonstrated performance improvements.
The baseline module keeps its default behaviour and its original acceptance policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from pi_jwm.step4_3b_dual_graph_encoder_v1 import (
    DualGraphEncoderConfig, PIJointGraphEncoder, TypeMLP,
)
from pi_jwm.step5_5_full_sharded_loader_v1 import FullFormalTrainer


SCHEMA_VERSION = "PI-JWM-Research-Geo-Temporal-Count-v1"
GEOMETRY_NAMES = [
    "physical_relation.delta_x_m", "physical_relation.delta_y_m",
    "physical_relation.delta_z_m", "physical_relation.distance_m",
    "physical_relation.relative_speed_mps", "physical_relation.relative_acceleration_mps2",
]


@dataclass(frozen=True)
class ResearchEncoderConfig(DualGraphEncoderConfig):
    enable_geometry_branch: bool = True
    enable_csi_history: bool = True
    enable_count_aggregation: bool = True


class GeoTemporalCountEncoder(PIJointGraphEncoder):
    def __init__(self, config: ResearchEncoderConfig, tensor_contract: Mapping[str, Any],
                 graph_contract: Mapping[str, Any], normalization_stats: Mapping[str, Any]):
        super().__init__(config, tensor_contract, graph_contract, normalization_stats)
        if not isinstance(config, ResearchEncoderConfig):
            raise TypeError("research encoder requires ResearchEncoderConfig")
        # Independent RNG context preserves initialization of all unchanged modules.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.initialization_seed + 10000)
            d, w = config.d_h, config.mlp_hidden_width
            if config.enable_geometry_branch:
                self.geometry_encoder = TypeMLP(12, w, d)
                self.p2c_gate = nn.Linear(4 * d, d)
                self.p2c_value = TypeMLP(4 * d, w, d)
            if config.enable_csi_history:
                self.comm_history_gru = nn.GRUCell(d, d)
                self.comm_history_fuse = TypeMLP(2 * d, w, d)
            if config.enable_count_aggregation:
                # Each family AND computational direction has its own processor.
                # Counts are relation incidences, not unique tasks or queue lengths.
                self.count_fuse = nn.ModuleDict({
                    f"{family}_{direction}": TypeMLP(d + 1, w, d)
                    for family in ("comm", "flow", "task_agent", "dag")
                    for direction in (("forward",) if family == "dag" else ("forward", "reverse"))
                })

    def geometry_features(self, tensor, graph):
        """Current endpoint differences; no physical-edge match is required.

        Speeds/accelerations are scalars, NOT radial relative velocities. Fixed
        train-only physical-relation scales are reused deliberately; no val fit.
        """
        device = next(self.parameters()).device
        get = lambda key, dtype: torch.as_tensor(tensor[key], dtype=dtype, device=device)[:, -1]
        geo = graph["blocks"]["geo_comm_relations"]
        src = torch.as_tensor(geo["source_physical_index"], dtype=torch.long, device=device)
        dst = torch.as_tensor(geo["target_physical_index"], dtype=torch.long, device=device)
        pos, dyn = get("entity_position_raw", torch.float32), get("entity_raw_features", torch.float32)
        pm, dm = get("entity_position_mask", torch.bool), get("entity_feature_mask", torch.bool)
        presence = get("entity_presence", torch.bool)
        endpoint = (src >= 0) & (dst >= 0) & (src < pos.shape[1]) & (dst < pos.shape[1])
        endpoint &= self._safe_gather(presence[..., None], src).squeeze(-1)
        endpoint &= self._safe_gather(presence[..., None], dst).squeeze(-1)
        endpoint &= torch.as_tensor(geo["presence"], dtype=torch.bool, device=device)
        endpoint &= torch.as_tensor(geo["validity"], dtype=torch.bool, device=device)
        delta = self._safe_gather(pos, dst) - self._safe_gather(pos, src)
        delta_dyn = self._safe_gather(dyn, dst) - self._safe_gather(dyn, src)
        pos_valid = self._safe_gather(pm, src) & self._safe_gather(pm, dst)
        dyn_valid = self._safe_gather(dm, src) & self._safe_gather(dm, dst)
        raw = torch.cat((delta, torch.linalg.vector_norm(delta, dim=-1, keepdim=True), delta_dyn), -1)
        mask = torch.cat((pos_valid, pos_valid.all(-1, keepdim=True), dyn_valid), -1)
        mask &= endpoint[..., None] & torch.isfinite(raw)
        raw = torch.where(mask, raw, torch.zeros_like(raw))
        return raw, mask

    def _p2c_context(self, tensor, graph, geo_src, geo_dst, comm_latent):
        original = super()._p2c_context(tensor, graph, geo_src, geo_dst, comm_latent)
        if not self.config.enable_geometry_branch:
            return original
        raw, mask = self.geometry_features(tensor, graph)
        encoded = self.geometry_encoder(torch.cat((self._normalize(raw, mask, GEOMETRY_NAMES), mask.float()), -1))
        encoded = encoded * mask.any(-1, keepdim=True)
        return torch.cat((original, encoded), -1)

    def audit_comm_history(self, tensor, comm):
        """Align by endpoint/type identity, never by time-varying raw row number.

        Formal v1 enumerates communication rows independently each frame. Local
        entity namespaces, unlike relation rows, are stable inside each window.
        Ambiguous duplicate links, malformed endpoints and RB order fail closed.
        """
        device = next(self.parameters()).device
        t = lambda name, dtype: torch.as_tensor(tensor[name], dtype=dtype, device=device)
        raw, mask = t("comm_csi_raw", torch.float32), t("comm_csi_mask", torch.bool)
        if raw.ndim != 4 or raw.shape != mask.shape or raw.shape[-1] != self.n_comm_rb:
            raise ValueError("CSI history must have aligned [batch, history, relation, RB] axes")
        current = torch.as_tensor(comm["csi"], dtype=torch.float32, device=device)
        if raw[:, -1].shape != current.shape:
            raise ValueError("CSI history/current relation widths differ")
        active = t("comm_relation_presence", torch.bool) & t("comm_relation_validity", torch.bool)
        current_active = torch.as_tensor(comm["presence"], dtype=torch.bool, device=device) & torch.as_tensor(comm["validity"], dtype=torch.bool, device=device)
        if not torch.equal(active[:, -1], current_active):
            raise ValueError("CSI history/current presence or validity mismatch")
        identity = []
        for key, graph_key in (("comm_source_index", "source_index"), ("comm_target_index", "target_index"),
                               ("comm_relation_type_index", "relation_type_index")):
            values = t(key, torch.long)
            reference = torch.as_tensor(comm[graph_key], dtype=torch.long, device=device)
            if bool(((values[:, -1] != reference) & current_active).any()):
                raise ValueError(f"CSI current relation identity mismatch: {key}")
            identity.append(values)
        expected_mask = torch.as_tensor(comm["csi_mask"], dtype=torch.bool, device=device)
        if not torch.equal(mask[:, -1], expected_mask) or not torch.equal(torch.where(mask[:, -1], raw[:, -1], 0), current):
            raise ValueError("CSI history/current values or masks mismatch")
        if not bool(torch.isfinite(raw[mask]).all()):
            raise ValueError("CSI history has nonfinite valid values")
        if bool((mask & ~active[..., None]).any()):
            raise ValueError("CSI feature mask is active on an invalid relation")
        rb, rb_mask = t("comm_rb_indices", torch.long), t("comm_rb_mask", torch.bool)
        canonical = torch.arange(self.n_comm_rb, device=device).view(1, 1, 1, -1)
        if bool(((rb != canonical) & rb_mask).any()) or bool((mask & ~rb_mask).any()):
            raise ValueError("CSI RB ordering is not canonical")
        src, dst, types = identity
        n_entity = int(tensor["entity_presence"].shape[-1])
        malformed = (src < 0) | (dst < 0) | (src >= n_entity) | (dst >= n_entity) | (types < 0) | (types >= 8)
        if bool((malformed & active).any()):
            raise ValueError("CSI history has invalid relation endpoints or type")
        # Keys are for lookup ONLY, and never become numeric neural features.
        keys = (src * n_entity + dst) * 8 + types
        sentinel = n_entity * n_entity * 8 + 1
        keys = torch.where(active, keys, sentinel)
        sorted_keys, order = keys.sort(dim=-1)
        if bool(((sorted_keys[..., 1:] == sorted_keys[..., :-1]) & (sorted_keys[..., 1:] != sentinel)).any()):
            raise ValueError("CSI history has ambiguous duplicate endpoint/type identities")
        queries = keys[:, -1:, :].expand_as(keys).contiguous()
        positions = torch.searchsorted(sorted_keys.contiguous(), queries).clamp(max=keys.shape[-1] - 1)
        matched = (sorted_keys.gather(-1, positions) == queries) & (queries != sentinel)
        historical_rows = order.gather(-1, positions)
        rb_rows = historical_rows[..., None].expand(*historical_rows.shape, self.n_comm_rb)
        aligned_mask = mask.gather(2, rb_rows) & matched[..., None]
        aligned_raw = torch.where(aligned_mask, raw.gather(2, rb_rows), torch.zeros_like(raw))
        return aligned_raw, aligned_mask, matched

    def _encode_comm(self, tensor, comm, csi, csi_mask, comm_type_idx):
        current = super()._encode_comm(tensor, comm, csi, csi_mask, comm_type_idx)
        if not self.config.enable_csi_history:
            return current
        raw, mask, active = self.audit_comm_history(tensor, comm)
        types = comm_type_idx[:, None].expand_as(active)
        # Missing/nonfinite inactive values are sanitized before any affine layer.
        sanitized = torch.where(mask, raw, torch.zeros_like(raw))
        csi_history = self._normalize(sanitized, mask, ["comm.channel_attenuation_db"] * self.n_comm_rb)
        encoded = self.comm_relation_encoder(torch.cat((csi_history, mask.float(), self.comm_type_embedding(types.clamp(0, 7))), -1))
        active = active & (types == 2) & mask.any(-1)
        hidden = self._presence_gru(encoded, active, self.comm_history_gru)
        fused = self.comm_history_fuse(torch.cat((current, hidden), -1))
        wireless = (comm_type_idx == 2) & active[:, -1]
        return torch.where(wireless[..., None], fused, current)

    def _aggregate(self, messages, indices, valid, count, family, direction):
        mean, has = super()._aggregate(messages, indices, valid, count, family, direction)
        if not self.config.enable_count_aggregation or family == "physical":
            return mean, has
        degree = messages.new_zeros((*mean.shape[:2], 1))
        for bi in range(messages.shape[0]):
            chosen = valid[bi] & (indices[bi] >= 0) & (indices[bi] < count)
            targets = indices[bi, chosen]
            degree[bi].index_add_(0, targets, messages.new_ones((targets.numel(), 1)))
        enriched = self.count_fuse[f"{family}_{direction}"](torch.cat((mean, torch.log1p(degree)), -1))
        # Critical: no-neighbour nodes must stay zero despite learned MLP biases.
        return enriched * has[..., None], has

    def forward(self, tensor, graph, *, validate_inputs=True):
        output = super().forward(tensor, graph, validate_inputs=validate_inputs)
        output["schema_version"] = SCHEMA_VERSION
        output["contract"]["experiment"] = {
            "geometry_branch": self.config.enable_geometry_branch,
            "csi_history": self.config.enable_csi_history,
            "count_aggregation": self.config.enable_count_aggregation,
            "geometry_scaling": "reuse_fixed_train_physical_relation_statistics",
            "count_semantics": "valid_relation_incidence_per_family_and_direction_not_unique_tasks",
            "baseline_acceptance_claim": False,
        }
        output["contract"]["message_passing"] = (
            "family_direction_masked_mean_log1p_count_residual_layernorm"
            if self.config.enable_count_aggregation else "family_specific_directed_masked_mean_residual_layernorm"
        )
        return output


class ResearchFullFormalTrainer(FullFormalTrainer):
    """The existing optimizer, shard loader, RSSM, losses and metrics are reused."""
    encoder_class = GeoTemporalCountEncoder


def save_research_encoder(model, path):
    torch.save({"schema_version": SCHEMA_VERSION, "config": asdict(model.config),
                "tensor_contract": model.tensor_contract, "graph_contract": model.graph_contract,
                "normalization_stats": model.normalization_stats, "state_dict": model.state_dict()}, Path(path))


def load_research_encoder(path):
    package = torch.load(Path(path), map_location="cpu", weights_only=False)
    if package.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("research encoder package schema mismatch")
    model = GeoTemporalCountEncoder(ResearchEncoderConfig(**package["config"]), package["tensor_contract"],
                                   package["graph_contract"], package["normalization_stats"])
    model.load_state_dict(package["state_dict"])
    return model
