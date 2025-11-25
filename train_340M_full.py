import os
import sys
import torch
import time
from pathlib import Path
import psutil

print("""
╔══════════════════════════════════════════════════════╗
║      340M Parameter NanoChat - FULL CPU Training     ║
║         This is the REAL DEAL - Days of training!    ║
╚══════════════════════════════════════════════════════╝
""")

# System info
print(f"\nSystem Information:")
print(f"  CPU: {psutil.cpu_count()} cores")
print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"  Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig

# CPU optimizations for maximum performance
num_threads = psutil.cpu_count(logical=False)  # Physical cores only
torch.set_num_threads(num_threads)
os.environ['OMP_NUM_THREADS'] = str(num_threads)
print(f"  Using {num_threads} CPU threads")

# Full nanochat d16 configuration (340M parameters)
config = GPTConfig(
    sequence_len=1024,   # Standard context
    vocab_size=50257,    # GPT-2 vocab
    n_layer=16,          # 16 layers (nanochat default)
    n_head=16,           # 16 attention heads
    n_kv_head=16,        # Standard attention
    n_embd=1024,         # 1024 embedding dimension
)

print(f"\nCreating FULL 340M parameter model...")
model = GPT(config).to('cpu')
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created: {total_params/1e6:.1f}M parameters")

# Memory check
mem_after_model = psutil.virtual_memory().available / (1024**3)
print(f"  RAM after model creation: {mem_after_model:.1f} GB available")

# Training configuration
batch_size = 1  # Minimum batch size for memory
seq_length = 128  # Short sequences for speed
learning_rate = 3e-4
num_iterations = 1000  # Start with 1000 for initial benchmark

# Optimizer
print("\nCreating optimizer...")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("✓ Optimizer ready")

# Checkpoint directory
checkpoint_dir = Path.home() / '.cache' / 'nanochat' / '340M_checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Benchmark first
print("\n" + "="*60)
print("BENCHMARKING SINGLE ITERATION...")
print("="*60)

x = torch.randint(0, 50257, (batch_size, seq_length))
y = torch.roll(x, -1, 1)

# Warmup
model.train()
for _ in range(3):
    loss = model(x, y)
    loss.backward()
    optimizer.zero_grad()

# Actual benchmark
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()

loss = model(x, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()

single_iter_time = time.time() - start

print(f"\nBenchmark Results:")
print(f"  Single iteration time: {single_iter_time:.2f} seconds")
print(f"  Tokens/sec: {batch_size * seq_length / single_iter_time:.1f}")
print(f"  Memory used: {psutil.Process().memory_info().rss / (1024**3):.1f} GB")

# Time estimates
estimated_hours_1k = (single_iter_time * 1000) / 3600
estimated_hours_10k = (single_iter_time * 10000) / 3600
estimated_hours_100k = (single_iter_time * 100000) / 3600
estimated_days_100k = estimated_hours_100k / 24

print(f"\nTime Estimates:")
print(f"  1,000 iterations: {estimated_hours_1k:.1f} hours")
print(f"  10,000 iterations: {estimated_hours_10k:.1f} hours ({estimated_hours_10k/24:.1f} days)")
print(f"  100,000 iterations (full): {estimated_hours_100k:.1f} hours ({estimated_days_100k:.1f} days)")

print("\n" + "="*60)
response = input("\nStart training? This will run for DAYS! (y/n): ")

if response.lower() == 'y':
    print(f"\n🚀 Starting 340M parameter training!")
    print(f"Checkpoints will save to: {checkpoint_dir}")
    print("You can stop anytime with Ctrl+C\n")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    try:
        for i in range(num_iterations):
            # Generate batch
            x = torch.randint(0, 50257, (batch_size, seq_length))
            y = torch.roll(x, -1, 1)
            
            # Forward/backward
            loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Logging
            if i % 10 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (i + 1) * batch_size * seq_length / elapsed
                eta_hours = (num_iterations - i) * elapsed / (i + 1) / 3600 if i > 0 else 0
                
                print(f"Iter {i:5d}/{num_iterations} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Tokens/s: {tokens_per_sec:.1f} | "
                      f"ETA: {eta_hours:.1f}h | "
                      f"RAM: {psutil.virtual_memory().percent:.1f}%")
            
            # Save checkpoint
            if i % 100 == 0 and i > 0:
                checkpoint_path = checkpoint_dir / f'model_340M_step_{i}.pt'
                torch.save({
                    'iteration': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': config,
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved: step {i}")
                
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving emergency checkpoint...")
        emergency_checkpoint = checkpoint_dir / f'model_340M_interrupted_{i}.pt'
        torch.save({
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item() if 'loss' in locals() else 0,
            'config': config,
        }, emergency_checkpoint)
        print(f"Saved to: {emergency_checkpoint}")
    
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training session complete!")
    print(f"  Time elapsed: {elapsed_total/3600:.2f} hours")
    print(f"  Final iteration: {i}")
    print(f"{'='*60}")
else:
    print("\nTraining cancelled. Benchmark data saved above.")