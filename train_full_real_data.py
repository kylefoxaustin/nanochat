import os
import sys
import torch
import time
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════╗
║     FULL NanoChat Training - Windows CPU Edition     ║
║            Using Real Training Data                  ║
╚══════════════════════════════════════════════════════╝
""")

# Import nanochat components
sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init

# Initialize compute (CPU mode)
compute_init(gpu_affinity=False, compile_enabled=False)

# CPU optimizations
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'

# Model configuration
config = GPTConfig(
    sequence_len=512,   # Shorter for CPU speed
    vocab_size=50257,   # GPT-2 vocab size
    n_layer=4,          # 4 layers (smaller for CPU)
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

# Create data loader
print("\nSetting up data loader...")
train_loader = tokenizing_distributed_data_loader(
    data_dir=str(Path.home() / '.cache' / 'nanochat' / 'base_data'),
    rank=0,
    world_size=1,
    batch_size=2,  # Very small for CPU
    sequence_len=512,
    model_copy_id=0,
    device='cpu'
)

# Training configuration
learning_rate = 3e-4
num_iterations = 5000  # Adjust based on your patience
save_every = 500
log_every = 10

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

print(f"\nTraining Configuration:")
print(f"  Learning rate: {learning_rate}")
print(f"  Total iterations: {num_iterations}")
print(f"  Save every: {save_every} steps")

# Checkpoint directory
checkpoint_dir = Path.home() / '.cache' / 'nanochat' / 'cpu_checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Training loop
print(f"\n{'='*60}")
print("Starting training with real data...")
print("Press Ctrl+C to stop and save checkpoint")
print(f"{'='*60}\n")

start_time = time.time()
model.train()
data_iter = iter(train_loader)

try:
    for iteration in range(num_iterations):
        # Get batch of real data
        try:
            x, y = next(data_iter)
        except StopIteration:
            # Restart data loader if we run out
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        
        # Move to CPU if needed
        x = x.to('cpu')
        y = y.to('cpu')
        
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
            tokens_per_sec = (iteration + 1) * 2 * 512 / elapsed if elapsed > 0 else 0
            
            # Estimate time remaining
            if iteration > 0:
                eta_seconds = elapsed * (num_iterations - iteration) / iteration
                eta_hours = eta_seconds / 3600
            else:
                eta_hours = 0
            
            print(f"Step {iteration:5d}/{num_iterations} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Tokens/s: {tokens_per_sec:.0f} | "
                  f"ETA: {eta_hours:.1f}h")
        
        # Save checkpoint
        if iteration % save_every == 0 and iteration > 0:
            checkpoint_path = checkpoint_dir / f'model_step_{iteration}.pt'
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: step {iteration}")

except KeyboardInterrupt:
    print("\n\nTraining interrupted! Saving checkpoint...")

# Final save
final_checkpoint = checkpoint_dir / f'model_final_{iteration}.pt'
torch.save({
    'iteration': iteration,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item() if 'loss' in locals() else 0,
}, final_checkpoint)

elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ Training session complete!")
print(f"  Total time: {elapsed_total/3600:.2f} hours")
print(f"  Steps completed: {iteration}/{num_iterations}")
print(f"  Model saved to: {final_checkpoint}")
print(f"  Model size: {total_params/1e6:.2f}M parameters")
print(f"{'='*60}")