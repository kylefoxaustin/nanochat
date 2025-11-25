import os
import sys
import torch
import time
from pathlib import Path
import numpy as np

print("""
╔══════════════════════════════════════════════════════╗
║     FULL NanoChat Training - Windows CPU Edition     ║
╚══════════════════════════════════════════════════════╝
""")

# Import nanochat components
sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import DataLoader

# CPU optimizations
torch.set_num_threads(8)  # Adjust based on your CPU cores
os.environ['OMP_NUM_THREADS'] = '8'

# Model configuration for CPU (smaller but real)
config = GPTConfig(
    sequence_len=1024,  # Standard context length
    vocab_size=50257,   # GPT-2 vocab size
    n_layer=6,          # 6 layers (small but real model)
    n_head=8,           # 8 attention heads
    n_kv_head=8,        # Standard attention
    n_embd=512,         # 512 embedding dimension
)

print(f"Model Configuration:")
print(f"  Layers: {config.n_layer}")
print(f"  Heads: {config.n_head}")
print(f"  Embedding dim: {config.n_embd}")
print(f"  Context length: {config.sequence_len}")

# Create model
model = GPT(config)
model = model.to('cpu')

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")

# Create data loader using your downloaded data
data_dir = Path.home() / '.cache' / 'nanochat' / 'base_data'
print(f"\nLoading data from: {data_dir}")

# Training configuration
batch_size = 4  # Small batch for CPU
learning_rate = 3e-4
num_iterations = 10000  # Adjust based on patience
save_every = 1000
log_every = 10

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

# Learning rate schedule
def get_lr(it):
    # Warmup for first 1000 steps
    if it < 1000:
        return learning_rate * it / 1000
    # Cosine decay
    if it > num_iterations * 0.8:
        return learning_rate * 0.1
    return learning_rate

print(f"\nTraining Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Total iterations: {num_iterations}")
print(f"  Save checkpoint every: {save_every} steps")

# Checkpoint directory
checkpoint_dir = Path.home() / '.cache' / 'nanochat' / 'cpu_checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Training loop
print(f"\n{'='*60}")
print("Starting training... (This will take a LONG time on CPU)")
print(f"{'='*60}\n")

start_time = time.time()
model.train()

for iteration in range(num_iterations):
    # Adjust learning rate
    lr = get_lr(iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Generate random batch (replace with real data loading when ready)
    seq_len = 256  # Shorter sequences for CPU speed
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    loss = model(x, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    
    # Logging
    if iteration % log_every == 0:
        elapsed = time.time() - start_time
        tokens_processed = (iteration + 1) * batch_size * seq_len
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        hours_elapsed = elapsed / 3600
        
        # Estimate time remaining
        if iteration > 0:
            total_estimated = elapsed * num_iterations / iteration
            hours_remaining = (total_estimated - elapsed) / 3600
        else:
            hours_remaining = 0
        
        print(f"Step {iteration:5d}/{num_iterations} | "
              f"Loss: {loss.item():.4f} | "
              f"LR: {lr:.2e} | "
              f"Tokens/sec: {tokens_per_sec:.1f} | "
              f"Time: {hours_elapsed:.1f}h | "
              f"ETA: {hours_remaining:.1f}h")
    
    # Save checkpoint
    if iteration % save_every == 0 and iteration > 0:
        checkpoint_path = checkpoint_dir / f'model_step_{iteration}.pt'
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'config': config,
        }, checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")

# Final save
final_checkpoint = checkpoint_dir / 'model_final.pt'
torch.save({
    'iteration': num_iterations,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'config': config,
}, final_checkpoint)

elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ Training Complete!")
print(f"  Total time: {elapsed_total/3600:.2f} hours")
print(f"  Final loss: {loss.item():.4f}")
print(f"  Model saved to: {final_checkpoint}")
print(f"{'='*60}")