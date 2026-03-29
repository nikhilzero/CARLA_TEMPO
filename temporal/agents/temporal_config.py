import os


class GlobalConfig:
    """Configuration for InterFuserTemporal closed-loop evaluation."""

    # Controller (identical to baseline)
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40

    max_throttle = 0.75
    brake_speed = 0.1
    brake_ratio = 1.1
    clip_delta = 0.35
    max_speed = 5
    collision_buffer = [2.5, 1.2]
    momentum = 0
    skip_frames = 1
    detect_threshold = 0.04

    # Temporal model settings
    model = "interfuser_temporal"   # signals temporal_agent.py to load InterFuserTemporal
    model_path = "/scratch/nd967/CARLA_TEMPO/runs/temporal/20260308-124902-temporal_research_T4_s5-T4/model_best.pth.tar"

    # Temporal parameters — must match training config
    temporal_frames = 4
    frame_stride = 5     # stride=5 at 10Hz → 2-second temporal context (DEC-005)
    temporal_depth = 2
