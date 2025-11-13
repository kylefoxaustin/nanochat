#!/bin/bash
# RTX 5090 single GPU training script - adapted from speedrun.sh
# Targets a sweet spot between the CPU demo and 8xH100 full run

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Setup environment (same as original)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# Rust BPE tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download data (reduced from 72 to ~36 for single GPU)
python -m nanochat.dataset -n 36 &

# Train tokenizer
python -m scripts.tok_train &
wait

# Train base model - depth=16 instead of 20, single GPU
python -m scripts.base_train \
    --depth=16 \
    --device_batch_size=8 \
    --total_batch_size=$((32 * 2048)) \
    --muon

# Midtraining
python -m scripts.mid_train \
    --device_batch_size=8

# SFT
python -m scripts.chat_sft \
    --device_batch_size=4

# Optional: Basic RL (commented out by default)
# python -m scripts.chat_rl --device_batch_size=2

# Generate report
python -m nanochat.report generate

echo "Training complete! Run 'python -m scripts.chat_web' to chat with your model"
