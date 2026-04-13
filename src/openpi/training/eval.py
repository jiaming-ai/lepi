"""Open-loop evaluation for training.

Periodically samples test episodes, picks a few timesteps per episode, runs
forward passes to get predicted action chunks, and overlays them on the
ground-truth trajectory. One plot per episode (subplots = joints, x-axis =
timestep across the full episode).
"""

import io
import logging
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

LEJU_JOINT_LABELS = [
    "L_joint_0", "L_joint_1", "L_joint_2", "L_joint_3",
    "L_joint_4", "L_joint_5", "L_joint_6", "L_gripper",
    "R_joint_0", "R_joint_1", "R_joint_2", "R_joint_3",
    "R_joint_4", "R_joint_5", "R_joint_6", "R_gripper",
]

# Colours for overlaying prediction chunks at different timesteps.
CHUNK_COLORS = ["red", "green", "purple", "orange", "cyan", "magenta", "brown", "pink"]


def _get_joint_labels(action_dim: int) -> list[str]:
    if action_dim <= len(LEJU_JOINT_LABELS):
        return LEJU_JOINT_LABELS[:action_dim]
    return [f"joint_{i}" for i in range(action_dim)]


# ---------------------------------------------------------------------------
# Setup helpers (called once at train start)
# ---------------------------------------------------------------------------

def create_eval_dataset(config: _config.TrainConfig) -> _data_loader.Dataset:
    """Create the raw (un-transformed) eval dataset for random access."""
    data_config = config.data.create(config.assets_dirs, config.model)
    return _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)


def create_eval_transforms(config: _config.TrainConfig):
    """Returns (data_config, fwd_transforms, pre_norm_transforms, unnorm, abs_transforms)."""
    data_config = config.data.create(config.assets_dirs, config.model)

    forward_transforms = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])

    pre_norm_transforms = _transforms.compose([
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ])

    unnorm_transform = _transforms.Unnormalize(
        data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
    )

    absolute_transforms = list(data_config.data_transforms.outputs)

    return data_config, forward_transforms, pre_norm_transforms, unnorm_transform, absolute_transforms


# ---------------------------------------------------------------------------
# Episode / timestep sampling (deterministic)
# ---------------------------------------------------------------------------

def _get_episode_boundaries(dataset: _data_loader.Dataset) -> tuple[dict[int, int], dict[int, int]]:
    inner = dataset._dataset if hasattr(dataset, "_dataset") else dataset
    if hasattr(inner, "_ep_starts") and hasattr(inner, "_ep_ends"):
        return dict(inner._ep_starts), dict(inner._ep_ends)
    if hasattr(inner, "meta") and hasattr(inner.meta, "episodes"):
        ep_starts, ep_ends = {}, {}
        for ep_info in inner.meta.episodes.values():
            ep_idx = ep_info["episode_index"]
            ep_starts[ep_idx] = ep_info.get("from", 0)
            ep_ends[ep_idx] = ep_info.get("to", len(inner))
        return ep_starts, ep_ends
    return {0: 0}, {0: len(inner)}


def _sample_eval_plan(
    dataset: _data_loader.Dataset,
    num_episodes: int,
    num_timesteps: int,
    action_horizon: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, list[int]]]:
    """Return [(ep_id, start, end, [sampled_indices]), ...].

    For each chosen episode, sample `num_timesteps` frame indices that are far
    enough from the episode end to produce a full action horizon chunk.
    """
    ep_starts, ep_ends = _get_episode_boundaries(dataset)
    episode_ids = sorted(ep_starts.keys())
    if not episode_ids:
        return []

    chosen = rng.choice(episode_ids, size=min(num_episodes, len(episode_ids)), replace=False)
    plan = []
    for eid in sorted(chosen):
        start, end = ep_starts[eid], ep_ends[eid]
        valid_end = max(start + 1, end - action_horizon)
        n_ts = min(num_timesteps, valid_end - start)
        ts = sorted(rng.choice(range(start, valid_end), size=n_ts, replace=False).tolist())
        plan.append((int(eid), start, end, ts))
    return plan


# ---------------------------------------------------------------------------
# Batched forward pass with explicit cleanup
# ---------------------------------------------------------------------------

EVAL_BATCH_SIZE = 32


def _batched_sample_actions(
    model_def: Any,
    params: at.Params,
    observations: list[dict],
    rng: jax.Array,
    num_steps: int = 10,
) -> np.ndarray:
    """Run sample_actions in sub-batches → [N, action_horizon, action_dim] numpy."""
    model = nnx.merge(model_def, params)
    model.eval()

    all_preds = []
    for i in range(0, len(observations), EVAL_BATCH_SIZE):
        chunk = observations[i : i + EVAL_BATCH_SIZE]
        batch = jax.tree.map(lambda *x: np.stack(x, axis=0), *chunk)
        obs = _model.Observation.from_dict(batch)
        obs = jax.device_put(obs)
        pred = model.sample_actions(rng, obs, num_steps=num_steps)
        all_preds.append(np.array(jax.device_get(pred)))
        del obs, pred

    del model
    jax.clear_caches()
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Unnormalize helpers
# ---------------------------------------------------------------------------

def _unnormalize_actions_batched(
    actions: np.ndarray,
    states: np.ndarray,
    unnorm_transform: _transforms.Unnormalize,
    absolute_transforms: list[_transforms.DataTransformFn],
) -> np.ndarray:
    """[N, ah, ad] normalised delta → [N, ah, ad] absolute."""
    out = []
    for i in range(actions.shape[0]):
        data = {"actions": actions[i], "state": states[i]}
        data = unnorm_transform(data)
        for t in absolute_transforms:
            data = t(data)
        out.append(data["actions"])
    return np.stack(out, axis=0)


def _get_gt_trajectory(
    dataset: _data_loader.Dataset,
    start: int,
    end: int,
    pre_norm_transforms: _transforms.DataTransformFn,
    unnorm_transform: _transforms.Unnormalize,
    absolute_transforms: list[_transforms.DataTransformFn],
    action_dim: int,
) -> np.ndarray:
    """Build GT action[0] trajectory over an episode → [T, action_dim].

    For each frame we take the first element of the action chunk (the action
    at that timestep) and convert to absolute joint positions.
    """
    gt = []
    for idx in range(start, end):
        raw = dict(dataset[idx])
        pre = pre_norm_transforms(dict(raw))
        state = np.asarray(pre["state"], dtype=np.float32)
        action_chunk = np.asarray(pre["actions"])  # [ah, ad] in delta space
        # Convert first action to absolute
        data = {"actions": action_chunk[:1], "state": state}
        for t in absolute_transforms:
            data = t(data)
        gt.append(data["actions"][0, :action_dim])
    return np.stack(gt, axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _make_episode_plot(
    gt_traj: np.ndarray,
    pred_chunks: list[tuple[int, np.ndarray]],
    action_dim: int,
    ep_id: int,
    ep_start: int,
    prefix: str,
) -> "wandb.Image":
    """One figure per episode.

    Args:
        gt_traj: [T, action_dim] ground truth absolute trajectory.
        pred_chunks: [(frame_idx, [ah, action_dim]), ...] predicted chunks.
        action_dim: meaningful joint dims.
        ep_id: episode id.
        ep_start: global index of episode start (to convert frame_idx to local).
        prefix: "ema" / "noema".
    """
    labels = _get_joint_labels(action_dim)
    T = gt_traj.shape[0]
    timesteps = np.arange(T)

    ncols = 4
    nrows = (action_dim + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)

    for j in range(action_dim):
        r, c = divmod(j, ncols)
        ax = axes[r][c]
        ax.plot(timesteps, gt_traj[:, j], "b-", linewidth=1.0, label="GT", zorder=1)

        for ci, (fidx, chunk) in enumerate(pred_chunks):
            local_start = fidx - ep_start
            ah = chunk.shape[0]
            xs = np.arange(local_start, local_start + ah)
            color = CHUNK_COLORS[ci % len(CHUNK_COLORS)]
            label = f"Pred t={local_start}" if j == 0 else None
            ax.plot(xs, chunk[:, j], "--", color=color, linewidth=1.2,
                    label=label, zorder=2)

        ax.set_title(labels[j], fontsize=9)
        ax.set_xlabel("timestep", fontsize=8)
        ax.set_ylabel("pos (abs)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Shared legend from first subplot
    handles, leg_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right", fontsize=8, ncol=2)

    for j in range(action_dim, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{prefix}  ep {ep_id}  ({T} steps, {len(pred_chunks)} chunks)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return wandb.Image(PILImage.open(buf))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def open_loop_eval(
    train_state,
    config: _config.TrainConfig,
    eval_dataset: _data_loader.Dataset,
    data_config: _config.DataConfig,
    forward_transforms: _transforms.DataTransformFn,
    pre_norm_transforms: _transforms.DataTransformFn,
    unnorm_transform: _transforms.Unnormalize,
    absolute_transforms: list[_transforms.DataTransformFn],
    step: int,
) -> dict[str, Any]:
    """Run open-loop evaluation.

    For each sampled episode:
      1. Build the GT absolute trajectory (cheap, no model call).
      2. At a few sampled timesteps, run the model → get predicted action chunks.
      3. Convert chunks to absolute, overlay on GT plot.
      4. Compute MSE / MAE on the predicted first-step actions.
    """
    eval_rng = np.random.default_rng(config.seed + 1)
    action_horizon = config.model.action_horizon

    plan = _sample_eval_plan(
        eval_dataset, config.eval_num_samples, config.eval_num_timesteps,
        action_horizon, eval_rng,
    )
    if not plan:
        logger.warning("No valid episodes found, skipping evaluation.")
        return {}

    total_preds = sum(len(ts) for _, _, _, ts in plan)
    logger.info(
        "Open-loop eval at step %d: %d episodes, %d forward passes",
        step, len(plan), total_preds,
    )

    # ------------------------------------------------------------------
    # 1. Collect model inputs for all sampled timesteps across episodes
    # ------------------------------------------------------------------
    all_transformed: list[dict] = []
    all_raw_states: list[np.ndarray] = []
    all_gt_norm: list[np.ndarray] = []  # for MSE/MAE computation

    for _, _, _, ts_indices in plan:
        for idx in ts_indices:
            raw = dict(eval_dataset[idx])
            pre = pre_norm_transforms(dict(raw))
            all_raw_states.append(np.asarray(pre["state"], dtype=np.float32))
            transformed = forward_transforms(dict(raw))
            all_gt_norm.append(np.asarray(transformed["actions"]))
            all_transformed.append(transformed)

    raw_states_np = np.stack(all_raw_states, axis=0)
    gt_norm_np = np.stack(all_gt_norm, axis=0)
    action_dim = raw_states_np.shape[-1]

    # ------------------------------------------------------------------
    # 2. Batched forward pass for all sampled timesteps (EMA + non-EMA)
    # ------------------------------------------------------------------
    jax_rng = jax.random.key(config.seed + 2)
    log_dict: dict[str, Any] = {}

    param_variants = [("noema", train_state.params)]
    if train_state.ema_params is not None:
        param_variants.append(("ema", train_state.ema_params))

    for variant_name, params in param_variants:
        pred_norm_np = _batched_sample_actions(
            train_state.model_def, params, all_transformed, jax_rng, num_steps=10,
        )

        # Unnormalize + delta → absolute  (all chunks)
        pred_abs_all = _unnormalize_actions_batched(
            pred_norm_np, raw_states_np, unnorm_transform, absolute_transforms,
        )
        gt_abs_all = _unnormalize_actions_batched(
            gt_norm_np, raw_states_np, unnorm_transform, absolute_transforms,
        )

        pred_abs_all = pred_abs_all[..., :action_dim]  # [N, ah, action_dim]
        gt_abs_all = gt_abs_all[..., :action_dim]

        # ------------------------------------------------------------------
        # 3. Metrics  (first-step actions across all sampled timesteps)
        # ------------------------------------------------------------------
        pred_first = pred_abs_all[:, 0, :]
        gt_first = gt_abs_all[:, 0, :]
        mse = float(np.mean((pred_first - gt_first) ** 2))
        mae = float(np.mean(np.abs(pred_first - gt_first)))
        per_joint_mse = np.mean((pred_first - gt_first) ** 2, axis=0)
        per_joint_mae = np.mean(np.abs(pred_first - gt_first), axis=0)

        prefix = f"eval_{variant_name}"
        log_dict[f"{prefix}/mse"] = mse
        log_dict[f"{prefix}/mae"] = mae
        joint_labels = _get_joint_labels(action_dim)
        for j in range(action_dim):
            log_dict[f"{prefix}/mse_{joint_labels[j]}"] = float(per_joint_mse[j])
            log_dict[f"{prefix}/mae_{joint_labels[j]}"] = float(per_joint_mae[j])

        # ------------------------------------------------------------------
        # 4. Per-episode plots
        # ------------------------------------------------------------------
        offset = 0
        plots = []
        for ep_id, ep_start, ep_end, ts_indices in plan:
            n_ts = len(ts_indices)

            # Build GT trajectory for this episode (cheap, no model call)
            gt_traj = _get_gt_trajectory(
                eval_dataset, ep_start, ep_end,
                pre_norm_transforms, unnorm_transform, absolute_transforms, action_dim,
            )

            # Collect predicted chunks for this episode
            ep_chunks = []
            for k in range(n_ts):
                fidx = ts_indices[k]
                chunk = pred_abs_all[offset + k]  # [ah, action_dim]
                ep_chunks.append((fidx, chunk))
            offset += n_ts

            plots.append(_make_episode_plot(
                gt_traj, ep_chunks, action_dim, ep_id, ep_start, prefix,
            ))

        log_dict[f"{prefix}/episode_plots"] = plots
        logger.info("Eval [%s] step %d: MSE=%.6f, MAE=%.6f", variant_name, step, mse, mae)

    return log_dict
