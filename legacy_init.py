"""Frozen module shapes retained for seed and checkpoint compatibility."""

import torch
import torch.nn as nn


class SpatiotemporalPatchSetEncoder(nn.Module):
    def __init__(
        self,
        patch_dim=512,
        embed_dim=256,
        patch_grid=(7, 7),
        max_steps=12,
        ff_dim=512,
        dropout=0.2,
        patch_dropout=0.0,
        frame_dropout=0.0,
    ):
        super().__init__()
        self.patch_grid = tuple(int(value) for value in patch_grid)
        if len(self.patch_grid) != 2 or min(self.patch_grid) < 1:
            raise ValueError("patch_grid must contain two positive integers")
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]
        self.max_steps = int(max_steps)
        self.patch_dropout = float(patch_dropout)
        self.frame_dropout = float(frame_dropout)
        if not 0 <= self.patch_dropout < 1:
            raise ValueError("patch_dropout must be in [0, 1)")
        if not 0 <= self.frame_dropout < 1:
            raise ValueError("frame_dropout must be in [0, 1)")

        self.projection = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
        )
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, self.max_steps, 1, embed_dim)
        )
        self.spatial_embedding = nn.Parameter(
            torch.zeros(1, 1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.trunc_normal_(self.spatial_embedding, std=0.02)
        self.refinement = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )


class CompetitiveSubgroupSlots(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_slots=6,
        num_iterations=2,
        ff_dim=512,
        dropout=0.2,
    ):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        self.embed_dim = int(embed_dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.scale = self.embed_dim**-0.5

        self.initial_slots = nn.Parameter(
            torch.empty(1, self.num_slots, self.embed_dim)
        )
        nn.init.trunc_normal_(self.initial_slots, std=0.02)
        self.slot_norm = nn.LayerNorm(self.embed_dim)
        self.member_norm = nn.LayerNorm(self.embed_dim)
        self.query = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.update = nn.GRUCell(self.embed_dim, self.embed_dim)
        self.refinement = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, self.embed_dim),
        )
        self.output_norm = nn.LayerNorm(self.embed_dim)


class SparseSubgroupGate(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=256, occupancy_bias=1.0):
        super().__init__()
        self.occupancy_bias = float(occupancy_bias)
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim * 2 + 1),
            nn.Linear(embed_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )


class QuerySequenceAggregator(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        sequence_length=7,
        num_heads=4,
        num_layers=1,
        ff_dim=512,
        dropout=0.2,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
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


__all__ = [
    "CompetitiveSubgroupSlots",
    "QuerySequenceAggregator",
    "SparseSubgroupGate",
    "SpatiotemporalPatchSetEncoder",
]
