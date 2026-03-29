"""
temporal_agent.py — CARLA leaderboard agent for InterFuserTemporal.

Wraps the baseline InterfuserAgent, replacing the single-frame forward pass
with a T-frame temporal forward pass using InterFuserTemporal.

The agent maintains a rolling buffer of the last T processed frame dicts.
When the buffer has fewer than T frames (start of episode), it duplicates
the most recent frame to fill the window — this mirrors how the temporal
model was trained (windows never cross episode boundaries).
"""

import os
import sys
import imp
import time
import math
import datetime
import pathlib

import cv2
import carla
import torch
import numpy as np
from PIL import Image
from collections import deque
from easydict import EasyDict
from torchvision import transforms

from leaderboard.autoagents import autonomous_agent
from team_code.utils import lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from team_code.interfuser_controller import InterfuserController
from team_code.render import render, render_self_car, render_waypoints
from team_code.tracker import Tracker

# ---- InterFuserTemporal import ----
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_INTERFUSER_PKG = os.path.join(_PROJECT_ROOT, "InterFuser", "interfuser")
for p in [_PROJECT_ROOT, _INTERFUSER_PKG]:
    if p not in sys.path:
        sys.path.insert(0, p)

from temporal.models.interfuser_temporal import build_interfuser_temporal

SAVE_PATH = os.environ.get("SAVE_PATH", "eval")
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)


def get_entry_point():
    return "TemporalAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        return pil_img.resize(self.size)


def create_carla_rgb_transform(input_size, need_scale=True,
                                mean=IMAGENET_DEFAULT_MEAN,
                                std=IMAGENET_DEFAULT_STD):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
        input_size_num = input_size[-1]
    else:
        img_size = input_size
        input_size_num = input_size

    tfl = []
    if need_scale:
        scale_map = {112: (170, 128), 128: (195, 146), 224: (341, 256), 256: (288, 288)}
        if input_size_num not in scale_map:
            raise ValueError(f"Unsupported input size: {input_size_num}")
        tfl.append(Resize2FixedSize(scale_map[input_size_num]))
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))
    return transforms.Compose(tfl)


class TemporalAgent(autonomous_agent.AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.initialized = False

        self.rgb_front_transform  = create_carla_rgb_transform(224)
        self.rgb_left_transform   = create_carla_rgb_transform(128)
        self.rgb_right_transform  = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.tracker = Tracker()
        self.softmax = torch.nn.Softmax(dim=1)

        self.config = imp.load_source("TemporalConfig", path_to_conf_file).GlobalConfig()
        self.skip_frames   = self.config.skip_frames
        self.num_frames    = self.config.temporal_frames
        self.frame_stride  = self.config.frame_stride
        self.controller    = InterfuserController(self.config)

        # Rolling buffer: stores the last (num_frames * frame_stride) processed frame dicts.
        # We sample every frame_stride-th entry to build the temporal window.
        self._frame_buffer = deque(maxlen=self.num_frames * self.frame_stride)

        # Load temporal model
        print(f"Loading InterFuserTemporal from: {self.config.model_path}")
        self.net = build_interfuser_temporal(
            num_frames=self.num_frames,
            temporal_encoder_depth=self.config.temporal_depth,
            pretrained_path=None,
        )
        ckpt = torch.load(self.config.model_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [WARN] Missing keys: {len(missing)}")
        self.net.cuda()
        self.net.eval()
        print(f"  Temporal model loaded. Params: {sum(p.numel() for p in self.net.parameters()):,}")

        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum   = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
            string += "_".join(map(lambda x: "%02d" % x,
                                   (now.month, now.day, now.hour, now.minute, now.second)))
            self.save_path = pathlib.Path(SAVE_PATH) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "width": 800, "height": 600, "fov": 100, "id": "rgb"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": -60.0,
             "width": 400, "height": 300, "fov": 100, "id": "rgb_left"},
            {"type": "sensor.camera.rgb", "x": 1.3, "y": 0.0, "z": 2.3,
             "roll": 0.0, "pitch": 0.0, "yaw": 60.0,
             "width": 400, "height": 300, "fov": 100, "id": "rgb_right"},
            {"type": "sensor.lidar.ray_cast", "x": 1.3, "y": 0.0, "z": 2.5,
             "roll": 0.0, "pitch": 0.0, "yaw": -90.0, "id": "lidar"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.05, "id": "imu"},
            {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "sensor_tick": 0.01, "id": "gps"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
        ]

    def tick(self, input_data):
        rgb       = cv2.cvtColor(input_data["rgb"][1][:, :, :3],       cv2.COLOR_BGR2RGB)
        rgb_left  = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3],  cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps     = input_data["gps"][1][:2]
        speed   = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if math.isnan(compass):
            compass = 0.0

        result = {"rgb": rgb, "rgb_left": rgb_left, "rgb_right": rgb_right,
                  "gps": gps, "speed": speed, "compass": compass}

        pos = self._get_position(result)

        lidar_data = input_data["lidar"][1]
        lidar_unprocessed = lidar_data[:, :3].copy()
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - compass, -pos[0], -pos[1],
            np.pi / 2 - compass, -pos[0], -pos[1],
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_command"] = next_cmd.value
        result["measurements"] = [pos[0], pos[1], compass, speed]

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point
        return result

    def _process_frame(self, tick_data):
        """Convert tick_data to the model input dict (same format as CarlaMVDetDataset)."""
        velocity = tick_data["speed"]
        command  = tick_data["next_command"]

        rgb = self.rgb_front_transform(
            Image.fromarray(tick_data["rgb"])).unsqueeze(0).cuda().float()
        rgb_left = self.rgb_left_transform(
            Image.fromarray(tick_data["rgb_left"])).unsqueeze(0).cuda().float()
        rgb_right = self.rgb_right_transform(
            Image.fromarray(tick_data["rgb_right"])).unsqueeze(0).cuda().float()
        rgb_center = self.rgb_center_transform(
            Image.fromarray(tick_data["rgb"])).unsqueeze(0).cuda().float()

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd_one_hot[command - 1] = 1
        cmd_one_hot.append(velocity)
        mes = torch.from_numpy(np.array(cmd_one_hot)).float().unsqueeze(0).cuda()

        return {
            "rgb":          rgb,
            "rgb_left":     rgb_left,
            "rgb_right":    rgb_right,
            "rgb_center":   rgb_center,
            "measurements": mes,
            "target_point": torch.from_numpy(
                tick_data["target_point"]).float().cuda().view(1, -1),
            "lidar": torch.from_numpy(
                tick_data["lidar"]).float().cuda().unsqueeze(0),
        }

    def _build_temporal_window(self):
        """
        Sample T frames from the buffer at interval frame_stride.
        If buffer has fewer than T*stride frames, pad by repeating the oldest.
        Returns list of T frame dicts (oldest → most recent).
        """
        buf = list(self._frame_buffer)  # oldest first
        # Sample indices: take every frame_stride-th frame counting back from latest
        indices = [max(0, len(buf) - 1 - i * self.frame_stride)
                   for i in range(self.num_frames - 1, -1, -1)]
        return [buf[i] for i in indices]

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)
        velocity  = tick_data["speed"]

        # Process current frame and push to buffer
        frame_dict = self._process_frame(tick_data)
        self._frame_buffer.append(frame_dict)

        # Build T-frame window and run temporal model
        window = self._build_temporal_window()   # list of T dicts
        (traffic_meta, pred_waypoints,
         is_junction, traffic_light_state,
         stop_sign, bev_feature) = self.net(window)

        traffic_meta        = traffic_meta.detach().cpu().numpy()[0]
        bev_feature         = bev_feature.detach().cpu().numpy()[0]
        pred_waypoints      = pred_waypoints.detach().cpu().numpy()[0]
        is_junction         = self.softmax(is_junction).detach().cpu().numpy().reshape(-1)[0]
        traffic_light_state = self.softmax(traffic_light_state).detach().cpu().numpy().reshape(-1)[0]
        stop_sign           = self.softmax(stop_sign).detach().cpu().numpy().reshape(-1)[0]

        if self.step % 2 == 0 or self.step < 4:
            traffic_meta = self.tracker.update_and_predict(
                traffic_meta.reshape(20, 20, -1),
                tick_data["gps"], tick_data["compass"], self.step // 2)
            traffic_meta = traffic_meta.reshape(400, -1)
            self.traffic_meta_moving_avg = (
                self.momentum * self.traffic_meta_moving_avg
                + (1 - self.momentum) * traffic_meta
            )
        traffic_meta = self.traffic_meta_moving_avg

        steer, throttle, brake, meta_infos = self.controller.run_step(
            velocity, pred_waypoints, is_junction,
            traffic_light_state, stop_sign, self.traffic_meta_moving_avg,
        )

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer    = float(steer)
        control.throttle = float(throttle)
        control.brake    = float(brake)

        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control

        if SAVE_PATH is not None:
            surround_map, _ = render(traffic_meta.reshape(20, 20, 7),
                                     pixels_per_meter=20)
            surround_map = surround_map[:400, 160:560]
            surround_map = np.stack([surround_map] * 3, 2)
            tick_data["surface"] = surround_map
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step // self.skip_frames
        surface = tick_data.get("surface")
        if surface is not None:
            Image.fromarray(surface.astype(np.uint8)).save(
                self.save_path / "meta" / ("%04d.jpg" % frame)
            )

    def destroy(self):
        del self.net
