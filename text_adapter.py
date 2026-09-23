"""Shared four-token Direct-M4 prompt adapter."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TEXT_DIM = 512
NUM_CLASSES = 3
PROMPTS_PER_CLASS = 3
NUM_CONTEXT_TOKENS = 4
TEXT_CONTEXT_SEED = 44003


def _validate_frozen_text_encoder(clip_model):
    text_prefixes = (
        "positional_embedding",
        "text_projection",
        "token_embedding.",
        "transformer.",
        "ln_final.",
    )
    trainable = [
        name
        for name, parameter in clip_model.named_parameters()
        if (name in text_prefixes or name.startswith(text_prefixes))
        and parameter.requires_grad
    ]
    if trainable:
        raise ValueError(
            "OpenCLIP text encoder must be frozen; trainable parameters include: "
            + ", ".join(trainable[:5])
        )


def encode_openclip_prompt_embeddings(
    clip_model,
    prompt_embeddings,
    tokenized_prompts,
):
    _validate_frozen_text_encoder(clip_model)
    if prompt_embeddings.dim() != 3:
        raise ValueError("prompt_embeddings must have shape [C, L, D_text]")
    if tokenized_prompts.shape != prompt_embeddings.shape[:2]:
        raise ValueError("tokenized_prompts must match prompt_embeddings [C, L]")
    context_length = int(clip_model.positional_embedding.size(0))
    if prompt_embeddings.size(1) != context_length:
        raise ValueError(
            f"Expected OpenCLIP context length {context_length}, got "
            f"{prompt_embeddings.size(1)}"
        )
    if getattr(clip_model, "text_pool_type", "argmax") != "argmax":
        raise ValueError("text_pool_type must be 'argmax'")

    cast_dtype = clip_model.transformer.get_cast_dtype()
    x = prompt_embeddings.to(dtype=cast_dtype)
    x = x + clip_model.positional_embedding.to(
        device=x.device,
        dtype=cast_dtype,
    )
    attention_mask = clip_model.attn_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(x.device)
    x = clip_model.transformer(x, attn_mask=attention_mask)
    x = clip_model.ln_final(x)

    tokenized_prompts = tokenized_prompts.to(x.device)
    positions = tokenized_prompts.argmax(dim=-1)
    batch_indices = torch.arange(x.size(0), device=x.device)
    pooled = x[batch_indices, positions]
    if clip_model.text_projection is None:
        raise ValueError("The OpenCLIP text encoder has no output projection")
    return F.normalize((pooled @ clip_model.text_projection).float(), dim=-1)


def _normalize(values):
    values = values.float()
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _flatten_prompts(prompts: Sequence[Sequence[str]]):
    if len(prompts) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, got {len(prompts)}")
    prompt_counts = {len(class_prompts) for class_prompts in prompts}
    if prompt_counts not in ({1}, {PROMPTS_PER_CLASS}):
        raise ValueError("Every class must contain exactly one or three prompts")
    flattened = tuple(prompt for group in prompts for prompt in group)
    if any(not isinstance(prompt, str) or not prompt for prompt in flattened):
        raise ValueError("All hard prompts must be non-empty strings")
    return flattened, prompt_counts.pop()


class SharedDirectM4TextAdapter(nn.Module):
    def __init__(
        self,
        clip_model,
        tokenizer,
        hard_prompts,
        fixed_prototypes,
        *,
        num_context_tokens=NUM_CONTEXT_TOKENS,
        context_seed=TEXT_CONTEXT_SEED,
        initialization_std=0.02,
    ):
        super().__init__()
        if int(num_context_tokens) != NUM_CONTEXT_TOKENS:
            raise ValueError("Direct-M4 requires exactly four context tokens")
        if float(initialization_std) <= 0:
            raise ValueError("initialization_std must be positive")
        if tuple(fixed_prototypes.shape) != (NUM_CLASSES, TEXT_DIM):
            raise ValueError(
                f"fixed_prototypes must have shape {(NUM_CLASSES, TEXT_DIM)}"
            )

        prompts, prompts_per_class = _flatten_prompts(hard_prompts)
        self._prompts_per_class = int(prompts_per_class)
        self.num_context_tokens = int(num_context_tokens)
        self.context_dim = int(clip_model.token_embedding.embedding_dim)
        self.context_length = int(clip_model.positional_embedding.size(0))
        if self.context_dim != TEXT_DIM or self.context_length != 77:
            raise ValueError("Direct-M4 requires a 77-token, 512-wide text encoder")

        original_tokens = tokenizer(list(prompts))
        expected_shape = (NUM_CLASSES * self._prompts_per_class, 77)
        if not torch.is_tensor(original_tokens):
            raise TypeError("OpenCLIP tokenizer must return a tensor")
        if tuple(original_tokens.shape) != expected_shape:
            raise ValueError(
                f"Text prompts must tokenize to shape {expected_shape}, got "
                f"{tuple(original_tokens.shape)}"
            )

        original_eot = original_tokens.argmax(dim=-1)
        suffix_length = self.context_length - 1 - self.num_context_tokens
        if int(original_eot.max()) + self.num_context_tokens >= self.context_length:
            raise ValueError("Context tokens would truncate a hard-prompt EOT")

        device = clip_model.token_embedding.weight.device
        with torch.no_grad():
            original_embeddings = clip_model.token_embedding(
                original_tokens.to(device)
            ).float()
        token_prefix = original_embeddings[:, :1].detach().clone()
        token_suffix = original_embeddings[:, 1 : 1 + suffix_length].detach().clone()

        placeholder_tokens = tokenizer([" ".join(["X"] * self.num_context_tokens)])
        if not torch.is_tensor(placeholder_tokens):
            raise TypeError("OpenCLIP tokenizer must return a tensor")
        if tuple(placeholder_tokens.shape) != (1, 77):
            raise ValueError("OpenCLIP tokenizer returned invalid context placeholders")
        placeholder_id = placeholder_tokens[0, 1].to(dtype=original_tokens.dtype)
        soft_tokens = torch.zeros_like(original_tokens)
        soft_tokens[:, 0] = original_tokens[:, 0]
        soft_tokens[:, 1 : 1 + self.num_context_tokens] = placeholder_id
        soft_tokens[:, 1 + self.num_context_tokens :] = original_tokens[
            :, 1 : 1 + suffix_length
        ]
        if not torch.equal(
            soft_tokens.argmax(dim=-1),
            original_eot + self.num_context_tokens,
        ):
            raise RuntimeError("Context insertion changed OpenCLIP EOT positions")

        self.register_buffer("token_prefix", token_prefix, persistent=False)
        self.register_buffer("token_suffix", token_suffix, persistent=False)
        self.register_buffer(
            "tokenized_prompts",
            soft_tokens.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "t_fixed",
            fixed_prototypes.float().detach().clone(),
            persistent=True,
        )

        self.shared_context = nn.Parameter(
            torch.empty(
                self.num_context_tokens,
                self.context_dim,
                device=device,
                dtype=torch.float32,
            )
        )
        # This isolated seed is part of the published initialization protocol.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(context_seed))
            nn.init.normal_(self.shared_context, std=float(initialization_std))
        self.register_buffer(
            "shared_context_init",
            self.shared_context.detach().clone(),
            persistent=True,
        )
        with torch.no_grad():
            t_soft_init = self._compute_soft_prototypes(clip_model)
        self.register_buffer(
            "t_soft_init",
            t_soft_init.detach().clone(),
            persistent=True,
        )

    @property
    def prompts_per_class(self):
        return self._prompts_per_class

    def _compute_soft_prototypes(self, clip_model):
        prefix = self.token_prefix.to(
            device=self.shared_context.device,
            dtype=self.shared_context.dtype,
        )
        suffix = self.token_suffix.to(
            device=self.shared_context.device,
            dtype=self.shared_context.dtype,
        )
        shared = self.shared_context.unsqueeze(0).expand(
            NUM_CLASSES * self.prompts_per_class,
            -1,
            -1,
        )
        prompt_features = encode_openclip_prompt_embeddings(
            clip_model,
            torch.cat((prefix, shared, suffix), dim=1),
            self.tokenized_prompts,
        )
        prompt_features = prompt_features.reshape(
            NUM_CLASSES,
            self.prompts_per_class,
            TEXT_DIM,
        )
        return _normalize(prompt_features.mean(dim=1))

    def forward(self, clip_model):
        return self._compute_soft_prototypes(clip_model)


__all__ = [
    "NUM_CONTEXT_TOKENS",
    "SharedDirectM4TextAdapter",
    "TEXT_CONTEXT_SEED",
]
