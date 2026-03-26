"""Custom dataset for Leju v3.0 format LeRobot datasets.

The openpi lerobot dependency uses v2.0/v2.1 format, but Leju datasets are in v3.0 format.
This module provides a custom Dataset class that reads v3.0 format directly, bypassing
the lerobot wrapper's version-specific expectations.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.common.datasets.video_utils import decode_video_frames

logger = logging.getLogger(__name__)


class LejuV3Dataset:
    """Dataset for Leju v3.0 LeRobot format with video and parquet data.

    Reads directly from v3.0 format files:
    - data/chunk-000/file-000.parquet (structured data: state, action, timestamps)
    - videos/{video_key}/chunk-000/file-{idx:03d}.mp4 (video frames)
    - meta/info.json, meta/tasks.parquet, meta/episodes/ (metadata)

    Args:
        root: Root directory of the dataset.
        delta_timestamps: Dict mapping keys to lists of relative timestamps for action chunking.
            Example: {"action": [0.0, 0.1, 0.2, ...]} for 10Hz data with action_horizon steps.
        tolerance_s: Tolerance in seconds for video frame matching.
        video_backend: Video decoding backend ("pyav", "torchcodec", etc.).
    """

    def __init__(
        self,
        root: str | Path,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 0.02,
        video_backend: str | None = None,
    ):
        self.root = Path(root)
        self.delta_timestamps = delta_timestamps or {}
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend

        # Load metadata
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)
        self.fps = self.info["fps"]

        # Identify video and non-video features
        self.video_keys = []
        self.data_features = []
        for key, feat in self.info["features"].items():
            if feat["dtype"] == "video":
                self.video_keys.append(key)
            elif feat["dtype"] in ("float32", "float64", "int64"):
                self.data_features.append(key)

        # Load tasks
        self.tasks = self._load_tasks()

        # Load parquet data
        self._load_parquet_data()

        # Load episode metadata for video file mapping
        self._load_episode_metadata()

        logger.info(
            "LejuV3Dataset loaded: %d frames, %d episodes, %d video keys",
            len(self), len(self.episode_meta), len(self.video_keys),
        )

    def _load_tasks(self) -> dict[int, str]:
        """Load task descriptions from tasks.parquet or tasks.jsonl."""
        tasks_parquet = self.root / "meta" / "tasks.parquet"
        tasks_jsonl = self.root / "meta" / "tasks.jsonl"

        if tasks_parquet.exists():
            table = pq.read_table(tasks_parquet)
            data = table.to_pydict()
            task_col = data.get("__index_level_0__", data.get("task", []))
            return {idx: task for idx, task in zip(data["task_index"], task_col)}
        elif tasks_jsonl.exists():
            tasks = {}
            with open(tasks_jsonl) as f:
                for line in f:
                    item = json.loads(line)
                    tasks[item["task_index"]] = item["task"]
            return tasks

        return {0: "Pick and Place"}

    def _load_parquet_data(self):
        """Load all parquet data files."""
        data_dir = self.root / "data"
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        tables = [pq.read_table(f) for f in parquet_files]
        self.table = tables[0] if len(tables) == 1 else pq.concat_tables(tables)
        self.num_frames = len(self.table)

        # Pre-extract episode boundaries for fast lookup
        ep_col = self.table.column("episode_index").to_pylist()
        self._ep_starts = {}
        self._ep_ends = {}
        for i, ep in enumerate(ep_col):
            if ep not in self._ep_starts:
                self._ep_starts[ep] = i
            self._ep_ends[ep] = i + 1  # exclusive

    def _load_episode_metadata(self):
        """Load episode metadata for video file mapping."""
        self.episode_meta = {}
        episodes_dir = self.root / "meta" / "episodes"

        parquet_files = sorted(episodes_dir.rglob("*.parquet")) if episodes_dir.exists() else []

        if parquet_files:
            for pf in parquet_files:
                table = pq.read_table(pf)
                data = table.to_pydict()
                for i in range(len(data["episode_index"])):
                    ep_idx = data["episode_index"][i]
                    ep_meta = {"episode_index": ep_idx}
                    for key in data:
                        if key != "episode_index":
                            ep_meta[key] = data[key][i]
                    self.episode_meta[ep_idx] = ep_meta
        else:
            # Fallback: assume single video file per camera
            ep_indices = sorted(self._ep_starts.keys())
            for ep_idx in ep_indices:
                self.episode_meta[ep_idx] = {"episode_index": ep_idx}

    def _get_video_path(self, ep_idx: int, vid_key: str) -> Path:
        """Get the video file path for a given episode and video key."""
        ep_meta = self.episode_meta.get(ep_idx, {})

        # v3.0 format: use chunk_index and file_index from episode metadata
        chunk_key = f"videos/{vid_key}/chunk_index"
        file_key = f"videos/{vid_key}/file_index"

        chunk_idx = ep_meta.get(chunk_key, 0)
        file_idx = ep_meta.get(file_key, 0)

        return self.root / "videos" / vid_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"

    def _get_video_timestamp(self, ep_idx: int, vid_key: str, frame_timestamp: float) -> float:
        """Convert a frame timestamp (relative to episode start) to video file timestamp."""
        ep_meta = self.episode_meta.get(ep_idx, {})
        from_ts_key = f"videos/{vid_key}/from_timestamp"
        from_ts = ep_meta.get(from_ts_key, 0.0)
        return from_ts + frame_timestamp

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx = self.num_frames + idx

        row = {col: self.table.column(col)[idx].as_py() for col in self.table.column_names}
        ep_idx = row["episode_index"]
        ep_start = self._ep_starts[ep_idx]
        ep_end = self._ep_ends[ep_idx]
        frame_in_ep = idx - ep_start

        item = {}

        # Extract structured data columns
        for col in self.data_features:
            val = row[col]
            if isinstance(val, list):
                item[col] = torch.tensor(val, dtype=torch.float32)
            elif isinstance(val, (int, float)):
                item[col] = torch.tensor(val)
            else:
                item[col] = val

        # Handle delta_timestamps: create action chunks by looking ahead
        for key, deltas in self.delta_timestamps.items():
            if key not in self.table.column_names:
                continue
            sequence = []
            for dt in deltas:
                target_idx = idx + round(dt * self.fps)
                # Clamp to episode boundary
                target_idx = max(ep_start, min(target_idx, ep_end - 1))
                val = self.table.column(key)[target_idx].as_py()
                if isinstance(val, list):
                    sequence.append(val)
                else:
                    sequence.append([val])
            item[key] = torch.tensor(sequence, dtype=torch.float32)

        # Decode video frames
        current_ts = frame_in_ep / self.fps
        for vid_key in self.video_keys:
            video_path = self._get_video_path(ep_idx, vid_key)
            video_ts = self._get_video_timestamp(ep_idx, vid_key, current_ts)
            try:
                frames = decode_video_frames(
                    video_path, [video_ts], self.tolerance_s, self.video_backend
                )
                item[vid_key] = frames.squeeze(0)  # Remove batch dim: (C, H, W)
            except Exception as e:
                logger.warning("Failed to decode %s frame for ep %d: %s", vid_key, ep_idx, e)
                shape = self.info["features"][vid_key]["shape"]
                item[vid_key] = torch.zeros(shape, dtype=torch.float32)

        # Add task info
        task_idx = row.get("task_index", 0)
        if isinstance(task_idx, list):
            task_idx = task_idx[0] if task_idx else 0
        item["task_index"] = torch.tensor(task_idx)
        item["task"] = self.tasks.get(task_idx, "Pick and Place")

        return item
