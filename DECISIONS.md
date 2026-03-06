# DECISIONS.md
# CARLA_TEMPO — Architectural Decisions and Reasoning

_Last updated: 2026-03-06_

---

## DEC-001 — Use InterFuser as the baseline model

**Decision**: Build on top of `opendilab/InterFuser` without forking or modifying upstream code.

**Rationale**:
- InterFuser is SOTA on the CARLA leaderboard at time of thesis start
- Multi-modal (RGB + LiDAR), multi-view, interpretable prediction heads
- Well-structured `timm`-based codebase with clean train/eval separation
- Pretrained checkpoints available; saves compute for baseline comparison

**Alternative considered**: Building from scratch, or using CARLA's own imitation learning agents.

**Risk**: InterFuser uses a fork of `timm` with custom additions. Updates to upstream timm may not be compatible.

---

## DEC-002 — Keep temporal/ isolated from InterFuser/

**Decision**: All temporal extension code lives in `temporal/`; `InterFuser/` is never modified (except known bug fixes in `train.py`).

**Rationale**:
- Clean separation of our contribution vs upstream baseline
- Easy to update InterFuser without merge conflicts
- Baseline remains runnable as-is for fair comparison
- `sys.path` manipulation used to import InterFuser's timm without installing it globally

**Alternative considered**: Fork InterFuser and modify it directly.

**Risk**: InterFuser's internal APIs (e.g., `forward_features()`) must remain stable. If upstream changes these, temporal model breaks.

---

## DEC-003 — Approach 1: Concat Fusion (first implementation)

**Decision**: Temporal fusion via concatenation: process each frame independently through the backbone, concatenate tokens across T, run a temporal Transformer encoder, pool back to N tokens.

**Rationale**:
- Lowest implementation risk — minimal changes to forward pass
- Temporal Transformer encoder is well-understood
- Can inherit all pretrained weights for the backbone portion
- Separates feature extraction from temporal reasoning cleanly

**Alternative considered (Approach 2)**: Temporal cross-attention — each frame's tokens attend to tokens from other frames during feature extraction. More expressive but harder to implement without modifying InterFuser internals.

**Status**: Approach 1 implemented and working. Approach 2 not yet started.

---

## DEC-004 — T=4 frames as default temporal window

**Decision**: Use T=4 consecutive frames as the default window size.

**Rationale**:
- At ~2 Hz capture rate: 4 frames = 2 seconds of history
- Long enough to capture vehicle dynamics (lane changes, deceleration, intersection approach)
- Short enough to fit 4× backbone forward passes in GPU memory (2× wall time vs baseline)
- Aligns with common choices in temporal video understanding literature (T=4–8)

**Alternative considered**: T=2 (too short for meaningful velocity estimation); T=8 (4× compute, memory risk).

**Ablation planned**: T=2, T=4, T=8 with same training setup (Phase 4).

---

## DEC-005 — frame_stride=1 for debug; stride=5 for thesis-main runs

**Decision**: Use `stride=1` during pipeline development and ablations on the debug subset.
For all thesis-quality training and evaluation runs on LMDrive data, use `stride=5`.

**Rationale**:
- The LMDrive dataset is collected at **10 Hz**
- The original InterFuser paper collected data at **2 Hz**
- At stride=1 with 10 Hz data: T=4 window covers 4 × 0.1s = **0.4 seconds** of history — far too short for meaningful velocity or intent estimation
- At stride=5 with 10 Hz data: T=4 window covers 4 × 0.5s = **2.0 seconds** of history — matches the intended 2-second temporal context from DEC-004
- All debug experiments (EXP-000–015) used stride=1 on 10 Hz data; results reflect 0.4s context, not 2s

**Concrete policy**:
| Run type | stride | Effective temporal context (T=4, 10 Hz) |
|---|---|---|
| Debug / smoke / ablation | 1 | 0.4 sec |
| **Thesis-main** | **5** | **2.0 sec** |

**Ablation planned**: stride=1 vs stride=5 vs stride=10 on research-scale data (Phase 4).

---

## DEC-006 — Pretrained backbone from 12-epoch baseline

**Decision**: Initialize the InterFuserTemporal backbone with the 12-epoch baseline checkpoint.

**Rationale**:
- The backbone already understands spatial features, object detection, waypoint prediction
- Fine-tuning the backbone with a lower LR (2e-4 vs 5e-4 for temporal params) preserves this knowledge
- Avoids training from scratch — saves GPU time and reduces risk of mode collapse
- Standard practice in video understanding (initialize from image-pretrained backbone)

**Alternative considered**: Training from random init or from timm pretrained weights.

---

## DEC-007 — Differential learning rates (backbone vs temporal)

**Decision**: Lower LR for pretrained backbone (2e-4), higher LR for new temporal parameters (5e-4).

**Rationale**:
- Prevents "catastrophic forgetting" of pretrained backbone weights
- Allows temporal components to learn quickly without overwriting backbone representations
- Same backbone_lr used in the InterFuser baseline for its pretrained backbone loading

---

## DEC-008 — Gradient accumulation (grad_accum=2, effective batch=4)

**Decision**: Use `batch_size=2` with `grad_accum=2` to achieve effective batch size 4 (same as baseline).

**Rationale**:
- Temporal model requires 4× compute per sample (T=4 forward passes per backbone)
- Raw batch_size=4 likely exceeds Amarel GPU VRAM (single RTX-class GPU)
- Gradient accumulation maintains matching statistics vs baseline without OOM errors

---

## DEC-009 — Loss function: same 5-head weighted sum as baseline

**Decision**: Use identical loss weighting as InterFuser baseline for temporal model.
`loss = traffic×0.5 + waypoints×0.2 + junction×0.05 + traffic_light×0.1 + stop_sign×0.01`

**Rationale**:
- Fair comparison with baseline — isolates the effect of temporal fusion
- Traffic and waypoints dominate (0.5+0.2=0.7) as the primary safety-relevant signals
- Exact loss functions copied verbatim to `temporal/utils/losses.py` to avoid subtle differences

---

## DEC-010 — Episode-aware temporal windows (no boundary crossing)

**Decision**: `TemporalWindowDataset` never creates windows that span two different routes/episodes.

**Rationale**:
- Each route is a continuous drive sequence; mixing frames from different routes would create nonsensical temporal context (teleporting between locations)
- `episode_lengths_from_carla_dataset()` reads `ds.route_frames` to derive boundaries
- Fallback: if `route_frames` not available, treat entire dataset as one episode

---

## DEC-011 — Use create_carla_loader for transforms (loader discarded)

**Decision**: Call `create_carla_loader()` to set transforms on the base dataset as a side effect, then discard the returned DataLoader.

**Rationale**:
- InterFuser's transform pipeline is complex (multi-view resizing, normalization, augmentation)
- `create_carla_loader` sets `rgb_transform`, `multi_view_transform`, etc. directly on the dataset object
- Duplicating this logic would be fragile and drift from baseline behavior
- Our own DataLoader (with `collate_temporal`) is then built on the transformed base dataset

**Risk**: This is undocumented behavior depending on `create_carla_loader`'s side effects. If InterFuser refactors this function, the side-effect may be lost.

---

## DEC-012 — CrossEntropyLoss for junction/traffic_light/stop_sign (NOT BCE)

**Decision**: Use `nn.CrossEntropyLoss()` for classification heads.

**Rationale**:
- Confirmed by inspecting the InterFuser baseline `train.py` directly
- Targets are integer class indices (0, 1, 2...) not one-hot floats
- BCE would require reformatting targets and would not match baseline behavior

---

## DEC-013 — Research-scale dataset scope (Towns + Weathers)

**Decision**: For thesis-quality experiments, download and train on:
- **Train towns**: Town01, Town02, Town03, Town04
- **Val/test town**: Town05 (held out — never seen during training)
- **Weather conditions**: [1, 3, 6, 8, 14, 18] (6 conditions covering daytime, rain, fog, overcast)
- **Route size**: `tiny` (start); upgrade to `short` if quota allows

**Rationale**:
- 4 training towns provides distributional diversity without requiring terabytes of storage
- Town05 is a natural generalization test (different road topology from Town01–04)
- 6 weather conditions cover the main visual variation; more conditions require proportionally more compute
- `tiny` routes have enough frames per route (~30–100) to support T=4 temporal windows

**Alternative considered**: All 8 towns + all 14 weathers (full LMDrive). Exceeds typical Amarel scratch quota (~500 GB) and would require multi-day download.

---

## OPEN DECISIONS (not yet made)

| ID | Question | Options | Status |
|---|---|---|---|
| OD-001 | What T gives best performance? | T=2, T=4, T=8 | UNKNOWN — ablation needed (on research-scale data) |
| OD-002 | Approach 1 vs Approach 2 for fusion? | Concat vs CrossAttn | UNKNOWN — Approach 2 not implemented |
| OD-003 | Is CARLA server available on Amarel for closed-loop eval? | Available / Not available | UNKNOWN — verify with `module avail` |
| OD-004 | Should we train on all available towns? | Town01-only vs multi-town | **DECIDED** → see DEC-013 |
| OD-005 | Use the new GitHub repo (when provided) fully, partially, or as reference? | Full / Partial / Reference | UNKNOWN — pending repo analysis |
