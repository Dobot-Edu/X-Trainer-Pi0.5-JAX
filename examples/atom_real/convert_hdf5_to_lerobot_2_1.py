"""
Convert Atom HDF5 recordings to LeRobot v2.1 format.

This converter builds a temporary Atom raw folder and then reuses
`examples/atom_real/convert_raw_to_lerobot_2_1.py` for final LeRobot export.

Expected HDF5 structure per episode:
  - action: (T, 28)
  - observations/qpos: (T, 28)
  - observations/images/top_left: (T, Nbytes)
  - observations/images/wrist_left: (T, Nbytes)
  - observations/images/wrist_right: (T, Nbytes)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import pickle
import shutil
import sys
import tempfile

import h5py
import numpy as np
from PIL import Image

try:
    from examples.atom_real import convert_raw_to_lerobot_2_1 as _atom_raw_converter
except ModuleNotFoundError:
    # Support running as a plain script path:
    # `python examples/atom_real/convert_hdf5_to_lerobot_2_1.py ...`
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from examples.atom_real import convert_raw_to_lerobot_2_1 as _atom_raw_converter


def _decode_jpeg_bytes(raw: np.ndarray) -> Image.Image:
    if raw.ndim != 1:
        raise ValueError(f"Expected encoded jpeg byte array with ndim=1, got {raw.shape}")
    data = raw.tobytes()
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")


def _find_hdf5_files(raw_root: Path) -> list[Path]:
    files = sorted(raw_root.rglob("*.hdf5")) + sorted(raw_root.rglob("*.h5"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under: {raw_root}")
    return files


def _build_temp_raw(hdf5_files: list[Path], temp_raw_root: Path) -> int:
    episode_count = 0

    for ep_index, path in enumerate(hdf5_files):
        with h5py.File(path, "r") as f:
            if "action" not in f or "observations/qpos" not in f:
                raise ValueError(f"Missing required datasets in {path}")

            action = np.asarray(f["action"])
            qpos = np.asarray(f["observations/qpos"])
            top = np.asarray(f["observations/images/top_left"])
            left = np.asarray(f["observations/images/wrist_left"])
            right = np.asarray(f["observations/images/wrist_right"])

            if action.ndim != 2 or qpos.ndim != 2:
                raise ValueError(f"Expected action/qpos rank 2 in {path}, got {action.shape}/{qpos.shape}")
            if action.shape[1] != 28 or qpos.shape[1] != 28:
                raise ValueError(f"Expected action/qpos dim 28 in {path}, got {action.shape}/{qpos.shape}")

            num_frames = min(action.shape[0], qpos.shape[0], top.shape[0], left.shape[0], right.shape[0])
            if num_frames <= 0:
                continue

            ep_dir = temp_raw_root / f"{ep_index:06d}"
            top_dir = ep_dir / "top_left"
            left_dir = ep_dir / "wrist_left"
            right_dir = ep_dir / "wrist_right"
            obs_dir = ep_dir / "observation"
            top_dir.mkdir(parents=True, exist_ok=True)
            left_dir.mkdir(parents=True, exist_ok=True)
            right_dir.mkdir(parents=True, exist_ok=True)
            obs_dir.mkdir(parents=True, exist_ok=True)

            for t in range(num_frames):
                _decode_jpeg_bytes(top[t]).save(top_dir / f"{t}.jpg", quality=95)
                _decode_jpeg_bytes(left[t]).save(left_dir / f"{t}.jpg", quality=95)
                _decode_jpeg_bytes(right[t]).save(right_dir / f"{t}.jpg", quality=95)

                payload = {
                    "obs": qpos[t].astype(np.float32),
                    "action": action[t].astype(np.float32),
                }
                with (obs_dir / f"{t}.pkl").open("wb") as fp:
                    pickle.dump(payload, fp)

            episode_count += 1

    return episode_count


def convert(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if output_root.exists() and args.overwrite_output:
        shutil.rmtree(output_root)

    hdf5_files = _find_hdf5_files(raw_root)

    with tempfile.TemporaryDirectory(prefix="atom_raw_", dir="/tmp") as tmp_dir:
        temp_raw_root = Path(tmp_dir) / "collect_data"
        temp_raw_root.mkdir(parents=True, exist_ok=True)

        converted_episodes = _build_temp_raw(hdf5_files, temp_raw_root)
        if converted_episodes <= 0:
            raise RuntimeError("No valid episodes were converted from HDF5 source.")

        passthrough = argparse.Namespace(
            raw_root=str(temp_raw_root),
            output_root=str(output_root),
            repo_id=args.repo_id,
            robot_type=args.robot_type,
            fps=args.fps,
            task=args.task,
            top_camera="top_left",
            use_videos=args.use_videos,
            vcodec=args.vcodec,
            repair_retries=args.repair_retries,
            encode_retries=args.encode_retries,
            encode_in_subprocess=args.encode_in_subprocess,
            quiet_encoder=args.quiet_encoder,
            keep_images_for_video=args.keep_images_for_video,
            skip_first_frames=args.skip_first_frames,
            max_episodes=0,
            max_frames_per_episode=0,
            min_frames=args.min_frames,
            skip_bad_frames=args.skip_bad_frames,
            overwrite_output=args.overwrite_output,
        )
        _atom_raw_converter.convert(passthrough)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Atom HDF5 data to LeRobot v2.1 format.")
    parser.add_argument("--raw_root", type=str, required=True, help="Root directory containing Atom .hdf5 episodes.")
    parser.add_argument("--output_root", type=str, required=True, help="Output root path for LeRobot v2.1 dataset.")
    parser.add_argument("--repo_id", type=str, default="local/dobot_atom_converted_v21")
    parser.add_argument("--robot_type", type=str, default="dobot_atom_upper")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", type=str, default="pick and place")
    parser.add_argument("--use_videos", dest="use_videos", action="store_true")
    parser.add_argument("--no_videos", dest="use_videos", action="store_false")
    parser.set_defaults(use_videos=True)
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--repair_retries", type=int, default=2)
    parser.add_argument("--encode_retries", type=int, default=1)
    parser.add_argument("--encode_in_subprocess", dest="encode_in_subprocess", action="store_true")
    parser.add_argument("--encode_in_process", dest="encode_in_subprocess", action="store_false")
    parser.set_defaults(encode_in_subprocess=True)
    parser.add_argument("--quiet_encoder", dest="quiet_encoder", action="store_true")
    parser.add_argument("--verbose_encoder", dest="quiet_encoder", action="store_false")
    parser.set_defaults(quiet_encoder=True)
    parser.add_argument("--keep_images_for_video", action="store_true")
    parser.add_argument("--skip_first_frames", type=int, default=0)
    parser.add_argument("--min_frames", type=int, default=10)
    parser.add_argument("--skip_bad_frames", dest="skip_bad_frames", action="store_true")
    parser.add_argument("--fail_on_bad_frames", dest="skip_bad_frames", action="store_false")
    parser.set_defaults(skip_bad_frames=True)
    parser.add_argument("--overwrite_output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
