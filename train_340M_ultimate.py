import os
import sys
import torch
import time
from pathlib import Path
import psutil

print("""
╔══════════════════════════════════════════════════════╗
║   304M Parameter NanoChat - Ultimate CPU Training    ║
║              Choose Your Training Level               ║
╚══════════════════════════════════════════════════════╝
""")

# System info
print(f"\nSystem Information:")
print(f"  CPU: {psutil.cpu_count()} cores")
print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"  Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig

# CPU optimizations
num_threads = psutil.cpu_count(logical=False)
torch.set_num_threads(num_threads)
os.environ['OMP_NUM_THREADS'] = str(num_threads)
print(f"  Using {num_threads} CPU threads")

# Training level selection
print("\n" + "="*60)
print("SELECT TRAINING LEVEL:")
print("="*60)
print("1. Quick Test      - 1,000 iterations   (~0.6 hours)")
print("2. Proof of Concept - 10,000 iterations  (~6 hours)")
print("3. Serious Training - 50,000 iterations  (~30 hours)")
print("4. FULL NANOCHAT   - 100,000 iterations (~60 hours)")
print("5. Custom          - Choose your own")
print("="*60)

choice = input("\nSelect level (1-5): ")

iteration_map = {
    '1': 1000,
    '2': 10000,
    '3': 50000,
    '4': 100000,
}

if choice == '5':
    num_iterations = int(input("Enter number of iterations: "))
elif choice in iteration_map:
    num_iterations = iteration_map[choice]
else:
    print("Invalid choice, defaulting to Quick Test (1000)")
    num_iterations = 1000

# Model configuration - Full 340M
config = GPTConfig(
    sequence_len=1024,
    vocab_size=50257,
    n_layer=16,
    n_head=16,
    n_kv_head=16,
    n_embd=1024,
)

print(f"\nCreating 304M parameter model...")
model = GPT(config).to('cpu')
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created: {total_params/1e6:.1f}M parameters")

# Training configuration
batch_size = 1
seq_length = 128
learning_rate = 3e-4
save_every = min(500, num_iterations // 10)  # Save 10 checkpoints total

# Optimizer
print("Creating optimizer...")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
print("✓ Optimizer ready")

# Checkpoint directory
checkpoint_dir = Path.home() / '.cache' / 'nanochat' / '340M_checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Check for existing checkpoints
existing_checkpoints = list(checkpoint_dir.glob('model_340M_step_*.pt'))
start_iteration = 0

if existing_checkpoints:
    print(f"\nFound {len(existing_checkpoints)} existing checkpoints")
    resume = input("Resume from latest checkpoint? (y/n): ")
    
    if resume.lower() == 'y':
        latest_checkpoint = max(existing_checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        print(f"✓ Resuming from iteration {start_iteration}")

# Time estimate
print("\nBenchmarking...")
x = torch.randint(0, 50257, (batch_size, seq_length))
y = torch.roll(x, -1, 1)

start = time.time()
loss = model(x, y)
loss.backward()
optimizer.zero_grad()
iter_time = time.time() - start

total_hours = (num_iterations - start_iteration) * iter_time / 3600
total_days = total_hours / 24

print(f"\n" + "="*60)
print(f"TRAINING PLAN:")
print(f"  Starting iteration: {start_iteration}")
print(f"  Target iterations: {num_iterations}")
print(f"  Iterations to go: {num_iterations - start_iteration}")
print(f"  Estimated time: {total_hours:.1f} hours ({total_days:.1f} days)")
print(f"  Checkpoints every: {save_every} iterations")
print(f"  Checkpoint location: {checkpoint_dir}")
print("="*60)

confirm = input(f"\nStart training? (y/n): ")

if confirm.lower() != 'y':
    print("Training cancelled.")
    sys.exit()

print(f"\n🚀 Starting 304M parameter training!")
print("Press Ctrl+C anytime to safely stop and save\n")
print("="*60 + "\n")

# Training loop
start_time = time.time()
model.train()

try:
    for i in range(start_iteration, num_iterations):
        # Generate batch
        x = torch.randint(0, 50257, (batch_size, seq_length))
        y = torch.roll(x, -1, 1)
        
        # Learning rate schedule (cosine decay)
        lr_mult = 0.5 * (1 + torch.cos(torch.tensor(3.14159 * i / num_iterations)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * lr_mult
        
        # Forward/backward
        loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Logging
        if i % 10 == 0:
            elapsed = time.time() - start_time
            iter_done = i - start_iteration + 1
            tokens_per_sec = iter_done * batch_size * seq_length / elapsed
            eta_hours = (num_iterations - i) * elapsed / iter_done / 3600 if iter_done > 0 else 0
            
            print(f"Iter {i:6d}/{num_iterations} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {learning_rate * lr_mult:.2e} | "
                  f"Tokens/s: {tokens_per_sec:.1f} | "
                  f"Progress: {100*i/num_iterations:.1f}% | "
                  f"ETA: {eta_hours:.1f}h")
        
        # Save checkpoint
        if i % save_every == 0 and i > 0:
            checkpoint_path = checkpoint_dir / f'model_340M_step_{i}.pt'
            torch.save({
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'config': config,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: step {i}")
            
            # Keep only last 5 checkpoints to save space
            all_checkpoints = sorted(checkpoint_dir.glob('model_340M_step_*.pt'))
            if len(all_checkpoints) > 5:
                for old_checkpoint in all_checkpoints[:-5]:
                    old_checkpoint.unlink()
                    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted! Saving checkpoint...")
    emergency_checkpoint = checkpoint_dir / f'model_340M_interrupted_{i}.pt'
    torch.save({
        'iteration': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if 'loss' in locals() else 0,
        'config': config,
    }, emergency_checkpoint)
    print(f"✓ Saved to: {emergency_checkpoint}")

# Final save
final_checkpoint = checkpoint_dir / f'model_340M_final_{i}.pt'
torch.save({
    'iteration': i,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item() if 'loss' in locals() else 0,
    'config': config,
}, final_checkpoint)

elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ Training Complete!")
print(f"  Total iterations: {i - start_iteration}")
print(f"  Time elapsed: {elapsed_total/3600:.2f} hours")
print(f"  Average tokens/sec: {(i - start_iteration) * batch_size * seq_length / elapsed_total:.1f}")
print(f"  Final loss: {loss.item() if 'loss' in locals() else 'N/A'}")
print(f"  Model saved to: {final_checkpoint}")
print(f"{'='*60}")
print(f"\n🎉 Your 304M parameter NanoChat model is ready!")