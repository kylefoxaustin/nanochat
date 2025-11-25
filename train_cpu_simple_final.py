import os
import sys
import torch
import time
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════╗
║     FULL NanoChat Training - Windows CPU Edition     ║
║              Simplified Version                      ║
╚══════════════════════════════════════════════════════╝
""")

sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig

# CPU optimizations
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'

# Model configuration
config = GPTConfig(
    sequence_len=512,
    vocab_size=50257,
    n_layer=4,
    n_head=8,
    n_kv_head=8,
    n_embd=512,
)

print(f"Model: {sum(p.numel() for p in GPT(config).parameters())/1e6:.1f}M parameters")

# Create model
model = GPT(config).to('cpu')

# Training config
B, T = 2, 256  # Batch size, sequence length
num_iters = 10000
learning_rate = 3e-4

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Checkpoint handling
checkpoint_dir = Path.home() / '.cache' / 'nanochat' / 'cpu_checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Resume if checkpoint exists
start_iter = 0
checkpoint_path = checkpoint_dir / 'latest.pt'
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_iter = checkpoint['iter']
    print(f"Resumed from iteration {start_iter}")

print(f"\nTraining {num_iters} iterations...")
print("="*60 + "\n")

start_time = time.time()
model.train()

for i in range(start_iter, num_iters):
    # Generate batch
    x = torch.randint(0, 50257, (B, T))
    y = torch.roll(x, -1, 1)
    
    # Forward, backward, optimize
    loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Log
    if i % 10 == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = (i - start_iter + 1) * B * T / elapsed
        eta_hours = (num_iters - i) * elapsed / (i - start_iter + 1) / 3600 if i > start_iter else 0
        print(f"Iter {i:5d}/{num_iters} | Loss: {loss.item():.4f} | "
              f"Tokens/s: {tokens_per_sec:.0f} | ETA: {eta_hours:.1f}h")
    
    # Save
    if i % 500 == 0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter': i
        }, checkpoint_path)
        print(f"  ✓ Checkpoint saved at iteration {i}")

print(f"\n✅ Training complete! Time: {(time.time()-start_time)/3600:.1f}h")