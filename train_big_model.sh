#!/bin/bash
echo "Starting 740M parameter model training!"
source .venv/bin/activate

# Base training - depth 24 with proper iterations
echo "Phase 1: Base training (depth=24)..."
python -m scripts.base_train \
    --depth=24 \
    --device_batch_size=4 \
    --total_batch_size=65536 \
    --num_iterations=20000

# Midtraining
echo "Phase 2: Midtraining..."
python -m scripts.mid_train \
    --device_batch_size=4

# SFT
echo "Phase 3: SFT (chat fine-tuning)..."
python -m scripts.chat_sft \
    --device_batch_size=2

# RL (Reinforcement Learning)
echo "Phase 4: RL (reinforcement learning)..."
python -m scripts.chat_rl \
    --device_batch_size=2 \
    --source=sft

# Generate report
python -m nanochat.report generate

echo "740M parameter model complete with RL! 🎉"
echo "Run 'python -m scripts.chat_web' to chat with your new AI!"
