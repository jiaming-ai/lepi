# Step 1: Compute norm stats (already done for both tasks)
  uv run scripts/compute_norm_stats.py --config-name pi05_leju_task1
  uv run scripts/compute_norm_stats.py --config-name pi05_leju_task2

  # Step 2: Train
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_leju_task1 --exp-name=my_exp --overwrite
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_leju_task2 --exp-name=my_exp --overwrite

  # Step 3: Serve trained policy
  uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=pi05_leju_task1 \
      --policy.dir=checkpoints/pi05_leju_task1/my_exp/30000