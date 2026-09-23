"""Frozen OpenCLIP encoder for full-frame and text features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenCLIPFrameEncoder(nn.Module):
    def __init__(
        self,
        backbone="ViT-B-16-quickgelu",
        pretrained="openai",
        freeze=True,
        unfreeze_visual_last_n_blocks=0,
        encode_batch_size=32,
        cache_dir=None,
    ):
        super().__init__()
        if not freeze or int(unfreeze_visual_last_n_blocks) != 0:
            raise ValueError("freeze must be true and visual blocks must stay frozen")
        if int(encode_batch_size) < 1:
            raise ValueError("encode_batch_size must be positive")
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for FrozenCLIPFrameEncoder"
            ) from exc

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            backbone,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )
        self.tokenizer = open_clip.get_tokenizer(backbone)
        self.encode_batch_size = int(encode_batch_size)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        visual = self.clip_model.visual
        if not hasattr(visual, "forward_intermediates"):
            raise TypeError(
                f"{type(visual).__name__} does not expose forward_intermediates"
            )
        if visual.proj is None:
            raise TypeError("The selected CLIP visual encoder has no output projection")
        self.grid_size = tuple(int(value) for value in visual.grid_size)
        self.output_dim = int(visual.proj.shape[-1])
        self.freeze = True
        self.text_frozen = True

    def forward(self, frames):
        if frames.dim() != 5 or frames.size(2) != 3:
            raise ValueError("frames must have shape [B, T, 3, H, W]")
        batch, steps, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch * steps, channels, height, width)
        scene_chunks = []
        with torch.no_grad():
            for start in range(0, flat_frames.size(0), self.encode_batch_size):
                images = flat_frames[start : start + self.encode_batch_size]
                outputs = self.clip_model.visual.forward_intermediates(
                    images,
                    indices=[-1],
                    normalize_intermediates=True,
                    output_fmt="NLC",
                )
                scene_chunks.append(outputs["image_features"].float())
        return torch.cat(scene_chunks, dim=0).reshape(batch, steps, self.output_dim)

    def encode_text_prompts(self, prompts):
        device = next(self.clip_model.parameters()).device
        tokens = self.tokenizer(list(prompts)).to(device)
        with torch.no_grad():
            embeddings = self.clip_model.encode_text(tokens).float()
        return F.normalize(embeddings, dim=-1)

    @staticmethod
    def trainable_state_dict():
        return {}


__all__ = ["FrozenCLIPFrameEncoder"]
