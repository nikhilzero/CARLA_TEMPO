"""
interfuser_temporal.py — Temporal extension of InterFuser.

Strategy: Run forward_features() for each of T consecutive frames,
add learnable temporal position embeddings, concatenate all tokens,
then pass through a temporal encoder before the original decoder.

This is a minimal, clean modification:
  - Reuses ALL of InterFuser's pretrained weights
  - Only adds temporal position embeddings + a small temporal encoder
  - The original encoder/decoder/heads are untouched
"""

import sys
import os
import torch
import torch.nn as nn

# Allow importing from InterFuser
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "InterFuser", "interfuser")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import timm


class InterFuserTemporal(nn.Module):
    """
    Temporal InterFuser: processes T consecutive frames and fuses their
    features before the transformer encoder+decoder.

    Architecture:
        For each frame t in [0, T):
            tokens_t = base_model.forward_features(frame_t)  # (N, B, D)
            tokens_t += temporal_pos_embed[t]                 # add temporal info
        
        all_tokens = concat(tokens_0, ..., tokens_{T-1})     # (T*N, B, D)
        memory = temporal_encoder(all_tokens)                 # fuse across time
        memory_pooled = temporal_pool(memory)                 # (N, B, D)
        
        # Then use original InterFuser decoder + heads
        hs = base_model.decoder(query_embed, memory_pooled)
        outputs = base_model.heads(hs, ...)

    Args:
        base_model: A fully constructed InterFuser (interfuser_baseline) instance.
        num_frames: Number of consecutive frames T (default 4).
        temporal_encoder_depth: Number of transformer layers for temporal fusion.
    """

    def __init__(
        self,
        base_model,
        num_frames: int = 4,
        temporal_encoder_depth: int = 2,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.base = base_model
        self.embed_dim = base_model.embed_dim

        # Learnable temporal position embedding: one per frame
        # Shape: (num_frames, 1, 1, embed_dim) -> broadcasts to (N, B, D)
        self.temporal_embed = nn.Parameter(
            torch.zeros(num_frames, 1, 1, self.embed_dim)
        )
        nn.init.normal_(self.temporal_embed, std=0.02)

        # Small temporal transformer encoder to fuse across frames
        # This attends over T*N tokens, allowing cross-frame attention
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=False,  # (seq, batch, dim) format like InterFuser
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_encoder_layer,
            num_layers=temporal_encoder_depth,
        )

        # Projection to pool T*N tokens back to N tokens
        # Simple approach: learned weighted average across temporal dimension
        self.temporal_pool = nn.Linear(num_frames, 1, bias=False)
        nn.init.constant_(self.temporal_pool.weight, 1.0 / num_frames)

    def forward(self, x_sequence):
        """
        Args:
            x_sequence: list of T dicts, each in InterFuser's expected format:
                {
                    "rgb": (B, 3, H, W),
                    "rgb_left": (B, 3, H, W),
                    "rgb_right": (B, 3, H, W),
                    "rgb_center": (B, 3, H, W),
                    "lidar": (B, 3, H, W),
                    "measurements": (B, M),
                    "target_point": (B, 2),
                }
        
        Returns:
            Same format as base InterFuser:
            (traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature)
        """
        assert len(x_sequence) == self.num_frames, (
            f"Expected {self.num_frames} frames, got {len(x_sequence)}"
        )

        # Step 1: Extract features from each frame using the base model's feature encoder
        all_frame_tokens = []
        for t, x in enumerate(x_sequence):
            # forward_features returns shape (N_tokens, B, D)
            tokens = self.base.forward_features(
                x["rgb"],
                x["rgb_left"],
                x["rgb_right"],
                x["rgb_center"],
                x["lidar"],
                x["measurements"],
            )
            # Add temporal position embedding for this frame
            # temporal_embed[t] shape: (1, 1, D), tokens shape: (N, B, D)
            tokens = tokens + self.temporal_embed[t]
            all_frame_tokens.append(tokens)

        # Step 2: Concatenate all frames' tokens along sequence dimension
        # Each tokens: (N, B, D) -> concatenated: (T*N, B, D)
        fused_tokens = torch.cat(all_frame_tokens, dim=0)

        # Step 3: Temporal encoder - cross-frame attention
        fused_tokens = self.temporal_encoder(fused_tokens)

        # Step 4: Pool back to original token count N
        # Reshape: (T*N, B, D) -> (T, N, B, D) -> (N, B, D, T) -> pool -> (N, B, D)
        N = all_frame_tokens[0].shape[0]
        B = all_frame_tokens[0].shape[1]
        T = self.num_frames

        fused_reshaped = fused_tokens.view(T, N, B, self.embed_dim)  # (T, N, B, D)
        fused_reshaped = fused_reshaped.permute(1, 2, 3, 0)  # (N, B, D, T)
        memory = self.temporal_pool(fused_reshaped).squeeze(-1)  # (N, B, D)

        # Step 5: Run through InterFuser's original spatial encoder
        memory = self.base.encoder(memory, mask=self.base.attn_mask)

        # Step 6: Decode using InterFuser's original decoder and prediction heads
        # Use the latest frame's measurements and target_point (most recent context)
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
            traffic_feature = hs[:, :400]
            is_junction_feature = hs[:, 400]
            traffic_light_state_feature = hs[:, 400]
            stop_sign_feature = hs[:, 400]
            waypoints_feature = hs[:, 401:411]
        else:
            traffic_feature = hs[:, :400]
            is_junction_feature = hs[:, 400]
            traffic_light_state_feature = hs[:, 400]
            stop_sign_feature = hs[:, 400]
            waypoints_feature = hs[:, 401:405]

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

        is_junction = self.base.junction_pred_head(is_junction_feature)
        traffic_light_state = self.base.traffic_light_pred_head(
            traffic_light_state_feature
        )
        stop_sign = self.base.stop_sign_head(stop_sign_feature)

        velocity = measurements[:, 6:7].unsqueeze(-1)
        velocity = velocity.repeat(1, 400, 32)
        traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)
        traffic = self.base.traffic_pred_head(traffic_feature_with_vel)

        return traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature


def build_interfuser_temporal(
    num_frames: int = 4,
    temporal_encoder_depth: int = 2,
    pretrained_path: str = None,
) -> InterFuserTemporal:
    """
    Build an InterFuserTemporal model.

    Args:
        num_frames: Temporal window size T.
        temporal_encoder_depth: Depth of temporal transformer encoder.
        pretrained_path: Path to a baseline InterFuser checkpoint to initialize from.
    """
    # Create the base InterFuser model
    base = timm.create_model("interfuser_baseline", pretrained=False)

    if pretrained_path is not None:
        assert os.path.exists(pretrained_path), f"Checkpoint not found: {pretrained_path}"
        state = torch.load(pretrained_path, map_location="cpu")
        state_dict = state.get("state_dict", state.get("model", state))
        missing, unexpected = base.load_state_dict(state_dict, strict=False)
        print(f"[build_interfuser_temporal] Loaded baseline from {pretrained_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model = InterFuserTemporal(
        base,
        num_frames=num_frames,
        temporal_encoder_depth=temporal_encoder_depth,
    )

    # Count new parameters
    base_params = sum(p.numel() for p in base.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    new_params = total_params - base_params
    print(f"[build_interfuser_temporal] Base params: {base_params:,}")
    print(f"[build_interfuser_temporal] Total params: {total_params:,}")
    print(f"[build_interfuser_temporal] New temporal params: {new_params:,} ({100*new_params/total_params:.1f}%)")

    return model