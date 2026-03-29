"""
interfuser_temporal_attn.py — Temporal InterFuser with cross-attention fusion (Approach 2).

Strategy: Use the most recent frame's tokens as queries and all past frames' tokens
as keys/values in a multi-head cross-attention block.

Compared to Approach 1 (concat fusion in interfuser_temporal.py):
  - More targeted: current frame explicitly selects what it needs from the past
  - Fewer parameters: no temporal pooling layer, no positional embeddings
  - Potentially better generalization: attention mechanism is inherently selective

Architecture:
    1. Extract tokens from each of T frames: base.forward_features(frame_t) → (N, B, D)
    2. Cross-attention: current_tokens (Q) attends to past_tokens (K, V)
       - current = tokens_{T-1}        shape: (N, B, D)
       - past    = cat(tokens_{0..T-2}) shape: ((T-1)*N, B, D)
    3. Repeat for `num_attn_layers` cross-attention blocks (each with residual + LN + FFN)
    4. Fused tokens → base.encoder(attn_mask) → decoder → 5 prediction heads
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "InterFuser", "interfuser")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import timm


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block: Q from current frame, K/V from past frames.
    Follows the standard pre-LN Transformer block structure:
        x = x + Dropout(CrossAttn(LN(x), LN(context)))
        x = x + Dropout(FFN(LN(x)))
    """

    def __init__(self, embed_dim: int, nhead: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm_q   = nn.LayerNorm(embed_dim)
        self.norm_kv  = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False,  # (seq, batch, dim) — matches InterFuser convention
        )

        ffn_dim = embed_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, current: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current: (N, B, D) — most-recent frame tokens (queries)
            past:    (M, B, D) — past frame tokens flattened (keys & values), M = (T-1)*N

        Returns:
            (N, B, D) — updated current-frame tokens
        """
        # Cross-attention with pre-LN
        q  = self.norm_q(current)
        kv = self.norm_kv(past)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        current = current + self.drop(attn_out)

        # FFN with pre-LN
        current = current + self.ffn(self.norm_ffn(current))
        return current


class InterFuserTemporalCrossAttn(nn.Module):
    """
    Temporal InterFuser using cross-attention fusion (Approach 2).

    Unlike Approach 1 which concatenates all frame tokens and runs a full
    temporal encoder, Approach 2 keeps the token count at N (same as the
    single-frame baseline) while still allowing the model to query the past.

    Args:
        base_model:       A fully constructed InterFuser (interfuser_baseline).
        num_frames:       Number of consecutive frames T (default 4).
        num_attn_layers:  Number of cross-attention blocks (default 2).
        dropout:          Dropout rate inside cross-attention blocks.
    """

    def __init__(
        self,
        base_model,
        num_frames: int = 4,
        num_attn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.base = base_model
        self.embed_dim = base_model.embed_dim

        # Learnable temporal position embedding for each frame
        # Shape (num_frames, 1, 1, D) broadcasts to (N, B, D) during feature extraction
        self.temporal_embed = nn.Parameter(
            torch.zeros(num_frames, 1, 1, self.embed_dim)
        )
        nn.init.normal_(self.temporal_embed, std=0.02)

        # Stack of cross-attention blocks
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=self.embed_dim,
                nhead=8,
                dropout=dropout,
                ffn_mult=4,
            )
            for _ in range(num_attn_layers)
        ])

    def forward(self, x_sequence):
        """
        Args:
            x_sequence: list of T dicts, each in InterFuser's expected format:
                {
                    "rgb":        (B, 3, H, W),
                    "rgb_left":   (B, 3, H, W),
                    "rgb_right":  (B, 3, H, W),
                    "rgb_center": (B, 3, H, W),
                    "lidar":      (B, C, H, W),
                    "measurements": (B, M),
                    "target_point": (B, 2),
                }
            Ordered oldest → most recent (x_sequence[-1] is the current frame).

        Returns:
            Same 6-tuple as base InterFuser:
            (traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature)
        """
        assert len(x_sequence) == self.num_frames, (
            f"Expected {self.num_frames} frames, got {len(x_sequence)}"
        )

        # ── Step 1: Extract per-frame tokens ──────────────────────────────────
        all_tokens = []
        for t, x in enumerate(x_sequence):
            tokens = self.base.forward_features(
                x["rgb"],
                x["rgb_left"],
                x["rgb_right"],
                x["rgb_center"],
                x["lidar"],
                x["measurements"],
            )  # (N, B, D)
            tokens = tokens + self.temporal_embed[t]  # add temporal pos embedding
            all_tokens.append(tokens)

        # ── Step 2: Cross-attention — current queries past ────────────────────
        current = all_tokens[-1]               # (N, B, D)
        if self.num_frames > 1:
            past = torch.cat(all_tokens[:-1], dim=0)  # ((T-1)*N, B, D)
            for layer in self.cross_attn_layers:
                current = layer(current, past)
        # If T=1, skip cross-attention (degenerate case — same as baseline)

        # `current` now holds temporally-enriched tokens: (N, B, D)

        # ── Step 3: Spatial encoder (InterFuser's original) ───────────────────
        memory = self.base.encoder(current, mask=self.base.attn_mask)

        # ── Step 4: Decode ────────────────────────────────────────────────────
        x = x_sequence[-1]
        front_image = x["rgb"]
        measurements = x["measurements"]
        target_point = x["target_point"]
        bs = front_image.shape[0]

        if self.base.end2end:
            tgt = self.base.query_pos_embed.repeat(bs, 1, 1)
        else:
            tgt = self.base.position_encoding(
                torch.ones((bs, 1, 20, 20), device=front_image.device)
            )
            tgt = tgt.flatten(2)
            tgt = torch.cat([tgt, self.base.query_pos_embed.repeat(bs, 1, 1)], 2)
        tgt = tgt.permute(2, 0, 1)

        hs = self.base.decoder(
            self.base.query_embed.repeat(1, bs, 1), memory, query_pos=tgt
        )[0]
        hs = hs.permute(1, 0, 2)  # (B, N_queries, D)

        if self.base.end2end:
            waypoints = self.base.waypoints_generator(hs, target_point)
            return waypoints

        if self.base.waypoints_pred_head != "heatmap":
            traffic_feature            = hs[:, :400]
            is_junction_feature        = hs[:, 400]
            traffic_light_state_feature = hs[:, 400]
            stop_sign_feature          = hs[:, 400]
            waypoints_feature          = hs[:, 401:411]
        else:
            traffic_feature            = hs[:, :400]
            is_junction_feature        = hs[:, 400]
            traffic_light_state_feature = hs[:, 400]
            stop_sign_feature          = hs[:, 400]
            waypoints_feature          = hs[:, 401:405]

        if self.base.waypoints_pred_head == "heatmap":
            waypoints = self.base.waypoints_generator(waypoints_feature, measurements)
        elif self.base.waypoints_pred_head == "gru":
            waypoints = self.base.waypoints_generator(waypoints_feature, target_point)
        elif self.base.waypoints_pred_head == "gru-command":
            waypoints = self.base.waypoints_generator(
                waypoints_feature, target_point, measurements
            )
        elif self.base.waypoints_pred_head == "linear":
            waypoints = self.base.waypoints_generator(waypoints_feature, measurements)
        elif self.base.waypoints_pred_head == "linear-sum":
            waypoints = self.base.waypoints_generator(waypoints_feature, measurements)

        is_junction         = self.base.junction_pred_head(is_junction_feature)
        traffic_light_state = self.base.traffic_light_pred_head(traffic_light_state_feature)
        stop_sign           = self.base.stop_sign_head(stop_sign_feature)

        velocity = measurements[:, 6:7].unsqueeze(-1)
        velocity = velocity.repeat(1, 400, 32)
        traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)
        traffic = self.base.traffic_pred_head(traffic_feature_with_vel)

        return traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature


def build_interfuser_temporal_crossattn(
    num_frames: int = 4,
    num_attn_layers: int = 2,
    dropout: float = 0.1,
    pretrained_path: str = None,
) -> InterFuserTemporalCrossAttn:
    """
    Build an InterFuserTemporalCrossAttn model (Approach 2).

    Args:
        num_frames:       Temporal window size T.
        num_attn_layers:  Number of cross-attention blocks.
        dropout:          Dropout rate in cross-attention.
        pretrained_path:  Path to a baseline InterFuser checkpoint.
    """
    base = timm.create_model("interfuser_baseline", pretrained=False)

    if pretrained_path is not None:
        assert os.path.exists(pretrained_path), f"Checkpoint not found: {pretrained_path}"
        state = torch.load(pretrained_path, map_location="cpu")
        state_dict = state.get("state_dict", state.get("model", state))
        missing, unexpected = base.load_state_dict(state_dict, strict=False)
        print(f"[build_interfuser_temporal_crossattn] Loaded baseline from {pretrained_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model = InterFuserTemporalCrossAttn(
        base,
        num_frames=num_frames,
        num_attn_layers=num_attn_layers,
        dropout=dropout,
    )

    base_params  = sum(p.numel() for p in base.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    new_params   = total_params - base_params
    print(f"[build_interfuser_temporal_crossattn] Base params:     {base_params:,}")
    print(f"[build_interfuser_temporal_crossattn] Total params:    {total_params:,}")
    print(f"[build_interfuser_temporal_crossattn] New attn params: {new_params:,} ({100*new_params/total_params:.1f}%)")

    return model
