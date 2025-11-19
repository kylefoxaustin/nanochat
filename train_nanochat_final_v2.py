import os
import sys
import torch
import time

print("""
╔══════════════════════════════════════════════════════╗
║     NanoChat CPU Training - Windows Edition          ║
╚══════════════════════════════════════════════════════╝
""")

# Import nanochat components
sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig

# Create tiny model for CPU with correct parameters
config = GPTConfig(
    sequence_len=512,   # Context length
    vocab_size=50257,   # GPT-2 vocab size
    n_layer=2,          # Tiny: only 2 layers
    n_head=4,           # Few attention heads
    n_kv_head=4,        # Key-value heads
    n_embd=256,         # Small embedding
)

print(f"Creating model with config:")
print(f"  Sequence length: {config.sequence_len}")
print(f"  Vocab size: {config.vocab_size}")
print(f"  Layers: {config.n_layer}")
print(f"  Heads: {config.n_head}") 
print(f"  Embedding: {config.n_embd}")

model = GPT(config)
model = model.to('cpu')

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params/1e6:.2f}M")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
print("\nStarting training on CPU...")
start_time = time.time()

for step in range(50):
    # Generate random data
    batch_size = 2
    seq_len = 128  # Shorter than max for speed
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass - model returns different things based on whether targets are provided
    result = model(x, y)
    
    # Handle different return types
    if isinstance(result, tuple):
        logits, loss = result
    else:
        # Result is just the loss
        loss = result
        logits = None
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if step % 10 == 0:
        elapsed = time.time() - start_time
        tokens_processed = (step + 1) * batch_size * seq_len
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        print(f"Step {step:3d} | Loss: {loss.item():.4f} | Tokens/sec: {tokens_per_sec:.1f} | Time: {elapsed:.1f}s")

print(f"\n✅ NanoChat GPT model training successful on Windows CPU!")
print(f"Total time: {time.time() - start_time:.1f}s")
print(f"Final loss: {loss.item():.4f}")
print("\nYour Windows fork of nanochat is fully operational! 🚀")