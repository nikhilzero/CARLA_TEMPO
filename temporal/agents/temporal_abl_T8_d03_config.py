import os

class GlobalConfig:
    """Configuration for ablation T=8 d03 (dropout=0.3) closed-loop evaluation.
    NOTE: model_path will be updated once training completes (run dir: 20260407-090828-ablation_T8_research_d03-T8).
    """
    turn_KP = 1.25; turn_KI = 0.75; turn_KD = 0.3; turn_n = 40
    speed_KP = 5.0; speed_KI = 0.5; speed_KD = 1.0; speed_n = 40
    max_throttle = 0.75; brake_speed = 0.1; brake_ratio = 1.1
    clip_delta = 0.35; max_speed = 5; collision_buffer = [2.5, 1.2]
    momentum = 0; skip_frames = 1; detect_threshold = 0.04
    model = "interfuser_temporal"
    model_path = "/scratch/nd967/CARLA_TEMPO/runs/ablations/20260407-090828-ablation_T8_research_d03-T8/model_best.pth.tar"
    temporal_frames = 8; frame_stride = 5; temporal_depth = 2
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)
