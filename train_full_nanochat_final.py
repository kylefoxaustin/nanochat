import os
import sys
import torch
import time
from pathlib import Path
import glob

print("""
╔══════════════════════════════════════════════════════╗
║     FULL NanoChat Training - Windows CPU Edition     ║
║         Complete Training with Real Data             ║
╚══════════════════════════════════════════════════════╝
""")

# Import nanochat components
sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.tokenizer import get_tokenizer

# CPU optimizations
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'

# Model configuration - optimized for CPU
config = GPTConfig(
    sequence_len=512,   # Context length
    vocab_size=50257,   # GPT-2 vocab size
    n_layer=4,          # 4 layers for CPU feasibility
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

# Get tokenizer
print("\nLoading tokenizer...")
try:
    tokenizer = get_tokenizer()
    print("✓ Tokenizer loaded")
except:
    print("Warning: Could not load tokenizer, will use default")
    tokenizer = None

# Create data loader with correct parameters
print("\nSetting up data loader...")
B = 2  # Batch size - small for CPU
T = 512  # Sequence length
split = 'train'

# Check for data files
data_dir = Path.home() / '.cache' / 'nanochat' / 'base_data'
data_files = list(data_dir.glob('*.parquet')) if data_dir.exists() else []
print(f"Found {len(data_files)} data files in {data_dir}")

try:
    train_loader = tokenizing_distributed_data_loader(
        B=B,
        T=T,
        split=split,
        tokenizer_threads=2,  # Fewer threads for CPU
        tokenizer_batch_size=32,  # Smaller batch for CPU
        device='cpu'  # CPU device
    )
    print("✓ Data loader created successfully with real data")
    using_real_data = True
except Exception as e:
    print(f"Warning: Could not create data loader: {e}")
    print("Will use synthetic data")
    using_real_data = False

# Training configuration
learning_rate = 3e-4
num_iterations = 10000  # Full training
save_every = 500
log_every = 10
eval_every = 100

# Learning rate schedule
def get_lr(it):
    # Warmup for first 1000 steps
    warmup_iters = 1000
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > num_iterations * 0.9:
        return learning_rate * 0.1
    # Cosine annealing
    decay_ratio = (it - warmup_iters) / (num_iterations - warmup_iters)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * decay_ratio)))
    return learning_rate * coeff

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

print(f"\nTraining Configuration:")
print(f"  Batch size: {B}")
print(f"  Sequence length: {T}")
print(f"  Learning rate: {learning_rate}")
print(f"  Total iterations: {num_iterations}")
print(f"  Using real data: {using_real_data}")

# Checkpoint directory
checkpoint_dir = Path.home() / '.cache' / 'nanochat' / 'cpu_checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Check for existing checkpoints to resume
existing_checkpoints = list(checkpoint_dir.glob('model_step_*.pt'))
if existing_checkpoints:
    latest_checkpoint = max(existing_checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    print(f"\nFound checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iteration = checkpoint['iteration'] + 1
    print(f"Resuming from iteration {start_iteration}")
else:
    start_iteration = 0
    print("\nStarting fresh training")

# Training loop
print(f"\n{'='*60}")
print("Starting full training...")
print("Press Ctrl+C to stop and save checkpoint")
print(f"{'='*60}\n")

start_time = time.time()
model.train()

if using_real_data:
    data_iter = iter(train_loader)

try:
    for iteration in range(start_iteration, num_iterations):
        # Adjust learning rate
        lr = get_lr(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        if using_real_data:
            try:
                batch = next(data_iter)
                if isinstance(batch, tuple):
                    x, y = batch
                else:
                    x = batch
                    y = torch.roll(x, shifts=-1, dims=1)
                x, y = x.to('cpu'), y.to('cpu')
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
                if isinstance(batch, tuple):
                    x, y = batch
                else:
                    x = batch
                    y = torch.roll(x, shifts=-1, dims=1)
                x, y = x.to('cpu'), y.to('cpu')
        else:
            # Synthetic data fallback
            x = torch.randint(0, config.vocab_size, (B, T))
            y = torch.roll(x, shifts=-1, dims=1)
        
        # Forward pass
        loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iteration % log_every == 0:
            elapsed = time.time() - start_time
            iter_elapsed = iteration - start_iteration + 1
            tokens_per_sec = iter_elapsed * B * T / elapsed if elapsed > 0 else 0
            
            # ETA calculation
            if iter_elapsed > 0:
                eta_seconds = elapsed * (num_iterations - iteration) / iter_elapsed
                eta_hours = eta_seconds / 3600
            else:
                eta_hours = 0
            
            print(f"Step {iteration:5d}/{num_iterations} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Grad: {grad_norm:.3f} | "
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
                'config': config,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: step {iteration}")
            
            # Clean up old checkpoints (keep only last 5)
            all_checkpoints = sorted(checkpoint_dir.glob('model_step_*.pt'))
            if len(all_checkpoints) > 5:
                for old_checkpoint in all_checkpoints[:-5]:
                    old_checkpoint.unlink()

except KeyboardInterrupt:
    print("\n\nTraining interrupted! Saving checkpoint...")

# Final save
final_checkpoint = checkpoint_dir / f'model_final_{iteration}.pt'
torch.save({
    'iteration': iteration,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item() if 'loss' in locals() else 0,
    'config': config,
}, final_checkpoint)

elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ Training session complete!")
print(f"  Total time: {elapsed_total/3600:.2f} hours")
print(f"  Steps completed: {iteration}/{num_iterations}")
print(f"  Final loss: {loss.item() if 'loss' in locals() else 'N/A'}")
print(f"  Model saved to: {final_checkpoint}")
print(f"  Model size: {total_params/1e6:.2f}M parameters")
print(f"{'='*60}")
print(f"\nYour nanochat model is ready!")
print(f"You've successfully trained a transformer on Windows CPU!")