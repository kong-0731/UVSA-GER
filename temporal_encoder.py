"""Scene-only temporal visual core for VGAF and GECV."""

import math

import torch
import torch.nn as nn

from .legacy_init import (
    CompetitiveSubgroupSlots,
    QuerySequenceAggregator,
    SparseSubgroupGate,
    SpatiotemporalPatchSetEncoder,
)


class PatchTemporalSceneEncoder(nn.Module):
    def __init__(
        self,
        input_dim=512,
        embed_dim=256,
        num_heads=4,
        num_layers=1,
        ff_dim=512,
        dropout=0.2,
        max_steps=12,
    ):
        super().__init__()
        self.max_steps = int(max_steps)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
        )
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, self.max_steps, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, scene_tokens):
        if scene_tokens.dim() != 3:
            raise ValueError("scene_tokens must have shape [B, T, D_scene]")
        batch, steps, _ = scene_tokens.shape
        if steps > self.max_steps:
            raise ValueError(f"Received {steps} steps but max_steps={self.max_steps}")
        frame_mask = torch.ones(
            batch,
            steps,
            dtype=torch.bool,
            device=scene_tokens.device,
        )
        tokens = self.projection(scene_tokens) + self.temporal_embedding[:, :steps]
        encoded = self.encoder(tokens, src_key_padding_mask=~frame_mask)
        mask_values = frame_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask_values).sum(dim=1) / mask_values.sum(dim=1)
        return self.output_norm(pooled)


class GlobalTemporalVisualCore(nn.Module):
    """Scene-only core with checkpoint-compatible parameters."""

    def __init__(
        self,
        patch_dim=512,
        scene_dim=512,
        embed_dim=256,
        output_dim=512,
        patch_grid=(7, 7),
        num_slots=6,
        slot_iterations=2,
        num_heads=4,
        relation_layers=1,
        fusion_layers=1,
        ff_dim=512,
        dropout=0.2,
        patch_dropout=0.0,
        frame_dropout=0.0,
        max_steps=12,
        branch_mode="scene_only",
        gate_mode="uniform",
    ):
        super().__init__()
        if branch_mode != "scene_only":
            raise ValueError("branch_mode must be 'scene_only'")
        if gate_mode != "uniform":
            raise ValueError("gate_mode must be 'uniform'")

        self.patch_grid = tuple(int(value) for value in patch_grid)
        self.embed_dim = int(embed_dim)
        self.output_dim = int(output_dim)
        self.branch_mode = branch_mode
        self.gate_mode = gate_mode

        # Frozen compatibility modules retain RNG consumption and state keys.
        self.patch_set_encoder = SpatiotemporalPatchSetEncoder(
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            patch_grid=self.patch_grid,
            max_steps=max_steps,
            ff_dim=ff_dim,
            dropout=dropout,
            patch_dropout=patch_dropout,
            frame_dropout=frame_dropout,
        )
        self.scene_encoder = PatchTemporalSceneEncoder(
            input_dim=scene_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=1,
            ff_dim=ff_dim,
            dropout=dropout,
            max_steps=max_steps,
        )
        self.subgroup_slots = CompetitiveSubgroupSlots(
            embed_dim=embed_dim,
            num_slots=num_slots,
            num_iterations=slot_iterations,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.subgroup_gate = SparseSubgroupGate(embed_dim=embed_dim)
        self.subgroup_aggregator = QuerySequenceAggregator(
            embed_dim=embed_dim,
            sequence_length=num_slots,
            num_heads=num_heads,
            num_layers=relation_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.visual_fusion = QuerySequenceAggregator(
            embed_dim=embed_dim,
            sequence_length=2,
            num_heads=num_heads,
            num_layers=fusion_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        self.patch_set_encoder.requires_grad_(False)
        self.subgroup_slots.requires_grad_(False)
        self.subgroup_gate.requires_grad_(False)
        self.subgroup_aggregator.requires_grad_(False)
        self.visual_fusion.requires_grad_(False)

    def forward(self, scene_tokens):
        return self.output_projection(self.scene_encoder(scene_tokens))


__all__ = ["GlobalTemporalVisualCore"]
