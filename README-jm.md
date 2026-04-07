
# For leju docker file add the following:
# # after active virtual env then install client
pip install openpi-client

curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 ~/.local/bin/uv sync
