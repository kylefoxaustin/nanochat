#!/bin/bash
# RTX 8000 FP32 training script - complete pipeline
# Optimized for 48GB VRAM with proven anti-NaN parameters

# Setup environment
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Install uv if needed
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# Install Rust and build tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download data (36 shards for depth=12)
python -m nanochat.dataset -n 36 &

# Train tokenizer
python -m scripts.tok_train &
wait

# Base training with RTX 8000 optimized parameters
# IMPORTANT: num_iterations=7080 keeps training time to ~60-70 hours
# Without this flag, it defaults to 56,640 iterations (17+ days!)
echo "Starting base training (depth=12, FP32)..."
python -m scripts.base_train \
    --depth=12 \
    --device_batch_size=1 \
    --total_batch_size=65536 \
    --num_iterations=7080 \
    --matrix_lr=0.005 \
    --embedding_lr=0.001 \
    --unembedding_lr=0.0001 \
    --grad_clip=0.5

# Midtraining
echo "Starting midtraining..."
python -m scripts.mid_train \
    --device_batch_size=1

# SFT (Supervised Fine-Tuning)
echo "Starting SFT..."
python -m scripts.chat_sft \
    --device_batch_size=1

# Generate report
python -m nanochat.report generate

echo "RTX 8000 training complete!"
echo "Run 'python -m scripts.chat_web' to chat with your model"
