"""
interfuser_temporal.py — Temporal extension of InterFuser.

Strategy: "concat" fusion (Approach 1, lowest risk).
  - Each frame is encoded independently by the same backbone.
  - Token sequences from T frames are concatenated along the sequence dimension.
  - The transformer encoder attends over all T*N tokens at once.
  - Temporal positional embeddings distinguish frames.

To switch to temporal attention (Approach 2), swap TemporalFusionConcat
with TemporalFusionAttention (to be implemented).
"""

import sys
import os
import torch
import torch.nn as nn

# Allow importing from InterFuser without installing as package
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "InterFuser", "interfuser")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from timm.models.interfuser import InterFuser


class TemporalPositionEmbedding(nn.Module):
    """Learnable per-frame position embedding added on top of spatial tokens."""

    def __init__(self, num_frames: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_frames, embed_dim)

    def forward(self, tokens: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        tokens: (B, N, D)
        frame_idx: integer index of this frame in the temporal window
        """
        t_embed = self.embed(
            torch.tensor(frame_idx, device=tokens.device)
        )  # (D,)
        return tokens + t_embed.unsqueeze(0).unsqueeze(0)


class InterFuserTemporal(nn.Module):
    """
    Temporal InterFuser: encodes T consecutive frames and fuses them
    before the transformer decoder.

    Args:
        base_model (InterFuser): Pre-built InterFuser instance (used as encoder).
        num_frames (int): Temporal window size T.
        fusion_strategy (str): 'concat' or 'temporal_attn' (concat implemented).
    """

    def __init__(
        self,
        base_model: InterFuser,
        num_frames: int = 4,
        fusion_strategy: str = "concat",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.fusion_strategy = fusion_strategy

        # Borrow sub-modules from the base InterFuser
        self.backbone_front = base_model.backbone_front
        self.backbone_left = base_model.backbone_left
        self.backbone_right = base_model.backbone_right
        self.backbone_rear = base_model.backbone_rear
        self.transformer = base_model.transformer
        self.decoder = getattr(base_model, "decoder", None)
        self.head = getattr(base_model, "head", None)

        # Retrieve embed dim from the backbone projection
        embed_dim = base_model.backbone_front.proj.out_channels

        # Temporal position embeddings (one per frame per camera)
        self.temporal_pos = TemporalPositionEmbedding(num_frames, embed_dim)

        if fusion_strategy == "concat":
            # Project concatenated temporal tokens back to original dim
            # (identity projection — no param overhead, just attention over more tokens)
            self.temporal_proj = nn.Identity()
        else:
            raise NotImplementedError(f"fusion_strategy='{fusion_strategy}' not yet implemented. Use 'concat'.")

    def encode_frame(self, front, left, right, rear, frame_idx: int):
        """
        Encode a single frame from all 4 cameras.
        Returns concatenated spatial tokens: (B, N_total, D)
        """
        f_feat, f_global = self.backbone_front(front)    # (B, D, H, W), (B, D, 1)
        l_feat, l_global = self.backbone_left(left)
        r_feat, r_global = self.backbone_right(right)
        b_feat, b_global = self.backbone_rear(rear)

        def flatten(x, g):
            # x: (B, D, H, W) -> (B, N, D);  g: (B, D, 1) -> (B, 1, D)
            return x.flatten(2).permute(0, 2, 1), g.permute(0, 2, 1)

        f_tok, f_g = flatten(f_feat, f_global)
        l_tok, l_g = flatten(l_feat, l_global)
        r_tok, r_g = flatten(r_feat, r_global)
        b_tok, b_g = flatten(b_feat, b_global)

        # Add temporal position embedding to each camera's tokens
        f_tok = self.temporal_pos(f_tok, frame_idx)
        l_tok = self.temporal_pos(l_tok, frame_idx)
        r_tok = self.temporal_pos(r_tok, frame_idx)
        b_tok = self.temporal_pos(b_tok, frame_idx)

        # Concatenate cameras: (B, 4*N + 4, D)  (+4 for global tokens)
        tokens = torch.cat([f_tok, f_g, l_tok, l_g, r_tok, r_g, b_tok, b_g], dim=1)
        return tokens

    def forward(self, frames: list, measurements: torch.Tensor = None):
        """
        Args:
            frames: list of T dicts, each with keys 'front', 'left', 'right', 'rear'
                    tensors of shape (B, 3, H, W).
            measurements: optional ego-motion / command tensor (B, M).
        Returns:
            Model output (waypoints / actions) — same format as base InterFuser.
        """
        assert len(frames) == self.num_frames, (
            f"Expected {self.num_frames} frames, got {len(frames)}"
        )

        # Encode each frame and concatenate along sequence dimension
        all_tokens = []
        for t, frame in enumerate(frames):
            tok = self.encode_frame(
                frame["front"], frame["left"], frame["right"], frame["rear"],
                frame_idx=t,
            )
            all_tokens.append(tok)

        # (B, T * (4*N + 4), D)
        fused = torch.cat(all_tokens, dim=1)
        fused = self.temporal_proj(fused)

        # Pass through transformer and decoder
        # NOTE: The exact API here depends on the InterFuser version.
        # Adjust this call once you inspect the full forward() of InterFuser.
        if self.decoder is not None:
            out = self.decoder(fused, measurements)
        else:
            out = fused  # placeholder

        return out


def build_interfuser_temporal(
    num_frames: int = 4,
    fusion_strategy: str = "concat",
    pretrained_path: str = None,
) -> InterFuserTemporal:
    """
    Build an InterFuserTemporal model.

    Args:
        num_frames: Temporal window size.
        fusion_strategy: 'concat' (only option for now).
        pretrained_path: Path to a baseline InterFuser checkpoint to warm-start from.
    """
    import timm

    base = timm.create_model("interfuser_baseline", pretrained=False)

    if pretrained_path is not None:
        import os
        assert os.path.exists(pretrained_path), f"Checkpoint not found: {pretrained_path}"
        state = torch.load(pretrained_path, map_location="cpu")
        # InterFuser checkpoints may wrap state under 'state_dict' or 'model'
        state_dict = state.get("state_dict", state.get("model", state))
        missing, unexpected = base.load_state_dict(state_dict, strict=False)
        print(f"[build_interfuser_temporal] Loaded backbone from {pretrained_path}")
        print(f"  Missing keys ({len(missing)}): {missing[:5]} ...")
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model = InterFuserTemporal(base, num_frames=num_frames, fusion_strategy=fusion_strategy)
    return model
