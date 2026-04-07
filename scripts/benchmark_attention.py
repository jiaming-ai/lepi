"""Benchmark JAX training step speed: baseline vs optimized attention.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/benchmark_attention.py
"""

import dataclasses
import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import common_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def init_train_state(config, init_rng, mesh):
    import flax.nnx as nnx
    import flax.traverse_util as traverse_util

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=None,
            ema_params=None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(init, in_shardings=replicated_sharding, out_shardings=state_sharding)(init_rng)
    return train_state, state_sharding


@at.typecheck
def train_step(config, rng, state, batch):
    import flax.nnx as nnx
    import optax

    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )

    info = {"loss": loss}
    return new_state, info


def create_fake_batch(config, rng, batch_size=None):
    """Create a fake batch matching the model's input spec."""
    if batch_size is None:
        batch_size = config.batch_size

    obs_spec, action_spec = config.model.inputs_spec(batch_size=batch_size)

    def make_array(spec):
        if spec.dtype == jnp.bool_:
            return jnp.ones(spec.shape, dtype=spec.dtype)
        elif jnp.issubdtype(spec.dtype, jnp.integer):
            return jax.random.randint(rng, spec.shape, 0, 100, dtype=spec.dtype)
        else:
            return jax.random.normal(rng, spec.shape, dtype=spec.dtype)

    observation = jax.tree.map(make_array, obs_spec)
    actions = make_array(action_spec)
    return observation, actions


def benchmark(config, num_warmup=5, num_steps=20, batch_size=None):
    """Run benchmark and return time per step."""
    if batch_size is None:
        batch_size = config.batch_size

    rng = jax.random.key(42)
    train_rng, init_rng, data_rng = jax.random.split(rng, 3)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    print(f"  Initializing model (batch_size={batch_size})...")
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)

    # Create fake batch
    batch = create_fake_batch(config, data_rng, batch_size)
    batch = jax.device_put(batch, data_sharding)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Warmup
    print(f"  Warming up ({num_warmup} steps)...")
    for i in range(num_warmup):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        jax.block_until_ready(train_state)

    # Benchmark
    print(f"  Benchmarking ({num_steps} steps)...")
    times = []
    for i in range(num_steps):
        start = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        jax.block_until_ready(train_state)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    loss = jax.device_get(info["loss"])
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times),
        "loss": float(loss),
        "times": times,
    }


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    config = _config.get_config("pi05_leju_task1")
    # Use smaller batch for faster benchmark
    config = config.__class__(**{
        **{f.name: getattr(config, f.name) for f in config.__dataclass_fields__.values()},
        "batch_size": 8,
        "wandb_enabled": False,
        "overwrite": True,
        "exp_name": "benchmark_tmp",
    })

    print("=" * 60)
    print("BENCHMARK: JAX Training Step")
    print(f"Config: pi05_leju_task1, batch_size={config.batch_size}")
    print("=" * 60)
    print()

    results = benchmark(config, num_warmup=3, num_steps=15, batch_size=config.batch_size)

    print()
    print(f"  Time per step: {results['mean']*1000:.1f} ± {results['std']*1000:.1f} ms")
    print(f"  Median: {results['median']*1000:.1f} ms")
    print(f"  Min/Max: {results['min']*1000:.1f} / {results['max']*1000:.1f} ms")
    print(f"  Loss: {results['loss']:.4f}")
    print()


if __name__ == "__main__":
    main()
