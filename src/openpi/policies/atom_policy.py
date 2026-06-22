import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


ATOM_STATE_DIM = 28


def make_atom_example() -> dict:
    """Creates a random input example for the Atom policy."""
    return {
        "observation.state": np.ones((ATOM_STATE_DIM,), dtype=np.float32),
        "observation.images.top": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AtomInputs(transforms.DataTransformFn):
    """Inputs for the Atom policy.

    Expected inputs:
    - observation.state: [28]
    - observation.images.top/left_wrist/right_wrist: image arrays
    - actions: [action_horizon, 28] (training only)
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation.state"])
        if state.shape[-1] < ATOM_STATE_DIM:
            raise ValueError(f"Expected state dim >= {ATOM_STATE_DIM}, got {state.shape}")

        images = self._extract_images(data)
        if set(images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected cameras in {self.EXPECTED_CAMERAS}, got {tuple(images)}")
        if "top" not in images:
            raise ValueError("Missing required camera: top")

        base_image = _parse_image(images["top"])
        left_wrist_image, left_mask = self._optional_image(images, "left_wrist", base_image)
        right_wrist_image, right_mask = self._optional_image(images, "right_wrist", base_image)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_mask,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs

    def _extract_images(self, data: dict) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        if "images" in data:
            images.update(data["images"])

        for camera_name in self.EXPECTED_CAMERAS:
            key = f"observation.images.{camera_name}"
            if key in data:
                images[camera_name] = data[key]
        return images

    def _optional_image(self, images: dict[str, np.ndarray], name: str, base_image: np.ndarray) -> tuple[np.ndarray, np.bool_]:
        if name not in images:
            return np.zeros_like(base_image), np.False_
        return _parse_image(images[name]), np.True_


@dataclasses.dataclass(frozen=True)
class AtomOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Return the [left_arm(7), left_hand(6), right_arm(7), right_hand(6), head(2)] action dimensions.
        return {"actions": np.asarray(data["actions"][:, :ATOM_STATE_DIM])}
