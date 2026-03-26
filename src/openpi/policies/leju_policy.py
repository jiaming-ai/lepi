"""Leju Kuavo4Pro robot policy transforms for OpenPI.

The Kuavo4Pro is a dual-arm humanoid robot with:
- State: [left_arm_7joints, gripper_l, right_arm_7joints, gripper_r] (16-dim)
- Action: [left_arm_7joints, left_claw, right_arm_7joints, right_claw] (16-dim)
- Cameras: head_cam_h (base), wrist_cam_l (left wrist), wrist_cam_r (right wrist)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

LEJU_ACTION_DIM = 16


def make_leju_example() -> dict:
    """Creates a random input example for the Leju policy (used for testing inference)."""
    return {
        "observation/state": np.random.rand(16).astype(np.float32),
        "observation/head_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "Pick and Place",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LejuInputs(transforms.DataTransformFn):
    """Maps Leju dataset format to the model input format.

    Expected input keys (after repack):
    - observation/state: [16] float32
    - observation/head_image: [H, W, 3] or [3, H, W] uint8
    - observation/wrist_image_left: [H, W, 3] or [3, H, W] uint8
    - observation/wrist_image_right: [H, W, 3] or [3, H, W] uint8
    - actions: [action_horizon, 16] float32 (only during training)
    - prompt: str
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)

        head_image = _parse_image(data["observation/head_image"])
        wrist_left = _parse_image(data["observation/wrist_image_left"])
        wrist_right = _parse_image(data["observation/wrist_image_right"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": wrist_left,
                "right_wrist_0_rgb": wrist_right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LejuOutputs(transforms.DataTransformFn):
    """Extracts Leju-specific actions from model output.

    The model outputs padded actions (32-dim); we return only the first 16 dims.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :LEJU_ACTION_DIM])}
