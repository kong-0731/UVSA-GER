"""UVSA-GER model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_encoder import FrozenCLIPFrameEncoder
from .temporal_encoder import GlobalTemporalVisualCore
from .text_adapter import (
    NUM_CONTEXT_TOKENS,
    TEXT_CONTEXT_SEED,
    SharedDirectM4TextAdapter,
)


DEFAULT_PATCH_PROMPTS = {
    "negative": (
        "a group of people showing negative emotion",
        "a group with an overall negative emotional atmosphere",
        "people collectively expressing negative affect",
    ),
    "neutral": (
        "a group of people showing neutral emotion",
        "a group with an overall neutral emotional atmosphere",
        "people collectively expressing neutral affect",
    ),
    "positive": (
        "a group of people showing positive emotion",
        "a group with an overall positive emotional atmosphere",
        "people collectively expressing positive affect",
    ),
}


class MeanAllFaceResidualBranch(nn.Module):
    def __init__(self, face_dim=512, hidden_dim=256):
        super().__init__()
        self.face_dim = int(face_dim)
        self.hidden_dim = int(hidden_dim)
        self.face_input_projection = nn.Sequential(
            nn.LayerNorm(self.face_dim),
            nn.Linear(self.face_dim, self.hidden_dim),
        )
        self.face_output_norm = nn.LayerNorm(self.hidden_dim)
        self.face_output_projection = nn.Linear(
            self.hidden_dim,
            self.face_dim,
            bias=False,
        )
        nn.init.zeros_(self.face_output_projection.weight)

    def forward(self, face_embeddings, face_mask):
        if face_embeddings.dim() != 4:
            raise ValueError(
                "face_embeddings must have shape [B, T, N, D_face]"
            )
        if face_mask.shape != face_embeddings.shape[:3]:
            raise ValueError(
                "face_mask must match face_embeddings dimensions [B, T, N]"
            )
        if face_embeddings.size(-1) != self.face_dim:
            raise ValueError(
                f"Expected face embedding dim {self.face_dim}, got "
                f"{face_embeddings.size(-1)}"
            )
        if face_embeddings.device != self.face_output_projection.weight.device:
            raise ValueError(
                "face_embeddings and the face branch must be on the same device"
            )

        face_mask = face_mask.to(device=face_embeddings.device, dtype=torch.bool)
        face_embeddings = face_embeddings.detach().to(
            dtype=self.face_output_projection.weight.dtype
        )
        safe_faces = torch.where(
            face_mask.unsqueeze(-1),
            face_embeddings,
            torch.zeros_like(face_embeddings),
        )
        faces_per_frame = face_mask.sum(dim=2)
        face_valid_mask = faces_per_frame > 0
        face_frame_raw = safe_faces.sum(dim=2) / faces_per_frame.clamp_min(1).to(
            face_embeddings.dtype
        ).unsqueeze(-1)
        face_hidden = self.face_input_projection(face_frame_raw)
        face_hidden = torch.where(
            face_valid_mask.unsqueeze(-1),
            face_hidden,
            torch.zeros_like(face_hidden),
        )
        valid_frame_count = face_valid_mask.sum(dim=1)
        face_video = face_hidden.sum(dim=1) / valid_frame_count.clamp_min(1).to(
            face_hidden.dtype
        ).unsqueeze(-1)
        face_video = self.face_output_norm(face_video)
        face_residual = self.face_output_projection(face_video)
        return face_residual * face_valid_mask.any(dim=1).to(
            face_residual.dtype
        ).unsqueeze(-1)


class UVSAGER(nn.Module):
    LABEL_ORDER = ("negative", "neutral", "positive")

    def __init__(
        self,
        backbone="ViT-B-16-quickgelu",
        pretrained="openai",
        freeze_clip=True,
        unfreeze_visual_last_n_blocks=0,
        clip_batch_size=32,
        clip_cache_dir=None,
        embed_dim=256,
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
        prompt_guided_slots=False,
        branch_mode="scene_only",
        gate_mode="uniform",
        *,
        text_context_seed=TEXT_CONTEXT_SEED,
        num_context_tokens=NUM_CONTEXT_TOKENS,
    ):
        super().__init__()
        if prompt_guided_slots:
            raise ValueError(
                "prompt_guided_slots must be false"
            )
        self.clip = FrozenCLIPFrameEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze=freeze_clip,
            unfreeze_visual_last_n_blocks=unfreeze_visual_last_n_blocks,
            encode_batch_size=clip_batch_size,
            cache_dir=clip_cache_dir,
        )
        self.core = GlobalTemporalVisualCore(
            patch_dim=self.clip.output_dim,
            scene_dim=self.clip.output_dim,
            embed_dim=embed_dim,
            output_dim=self.clip.output_dim,
            patch_grid=self.clip.grid_size,
            num_slots=num_slots,
            slot_iterations=slot_iterations,
            num_heads=num_heads,
            relation_layers=relation_layers,
            fusion_layers=fusion_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            patch_dropout=patch_dropout,
            frame_dropout=frame_dropout,
            max_steps=max_steps,
            branch_mode=branch_mode,
            gate_mode=gate_mode,
        )
        self.prompt_bank = {
            label: DEFAULT_PATCH_PROMPTS[label] for label in self.LABEL_ORDER
        }
        self.register_buffer("_label_cache", torch.empty(0), persistent=False)

        if self.core.branch_mode != "scene_only":
            raise ValueError("Mean-all fusion requires branch_mode='scene_only'")
        if self.core.embed_dim != 256 or self.core.output_dim != 512:
            raise ValueError(
                "Mean-all fusion requires embed_dim=256 and output_dim=512"
            )
        if self.clip.output_dim != 512:
            raise ValueError("Mean-all fusion requires 512-D CLIP face embeddings")
        self.core.face_branch = MeanAllFaceResidualBranch(
            face_dim=self.core.output_dim,
            hidden_dim=self.core.embed_dim,
        )

        fixed = self.encode_label_embeddings().detach().float().clone()
        self.core.shared_direct_m4_text_adapter = SharedDirectM4TextAdapter(
            clip_model=self.clip.clip_model,
            tokenizer=self.clip.tokenizer,
            hard_prompts=self.prompt_bank.values(),
            fixed_prototypes=fixed,
            num_context_tokens=num_context_tokens,
            context_seed=text_context_seed,
        )

    @property
    def direct_text(self):
        return self.core.shared_direct_m4_text_adapter

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def encode_label_embeddings(self):
        if self._label_cache.numel() > 0:
            return self._label_cache
        prompts = [
            prompt
            for label in self.LABEL_ORDER
            for prompt in self.prompt_bank[label]
        ]
        embeddings = self.clip.encode_text_prompts(prompts)
        embeddings = embeddings.reshape(len(self.LABEL_ORDER), -1, embeddings.size(-1))
        self._label_cache = F.normalize(embeddings.mean(dim=1), dim=-1).detach()
        return self._label_cache

    def forward(self, frames, face_embeddings, face_mask):
        if frames.shape[:2] != face_embeddings.shape[:2]:
            raise ValueError(
                "Global frames and cached faces must use the same [B, T] positions"
            )
        global_raw = self.core(self.clip(frames))
        face_residual = self.core.face_branch(face_embeddings, face_mask).to(
            global_raw.dtype
        )
        fused_raw = global_raw + face_residual
        text_prototypes = self.direct_text(self.clip.clip_model)
        fused_feature = F.normalize(fused_raw.float(), dim=-1)
        scale = self.core.logit_scale.exp().clamp(max=100)
        return {"logits": scale * (fused_feature @ text_prototypes.t())}


__all__ = ["UVSAGER"]
