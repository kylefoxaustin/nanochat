import os
import sys
import torch
from pathlib import Path
import time

print("""
╔══════════════════════════════════════════════════════╗
║     NanoChat CPU Training - Windows Edition          ║
╚══════════════════════════════════════════════════════╝
""")

# Import nanochat components
sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig

# Create tiny model for CPU
config = GPTConfig(
    vocab_size=50257,
    n_layer=2,      # Tiny: only 2 layers
    n_head=4,       # Few attention heads
    n_embd=256,     # Small embedding
    block_size=512, # Short context
    bias=False,
    dropout=0.0,
)

print(f"Creating model with config:")
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
print("\nStarting training...")
start_time = time.time()

for step in range(50):
    # Generate random data (bypass tokenizer)
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits, loss = model(x, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if step % 10 == 0:
        elapsed = time.time() - start_time
        tokens_processed = (step + 1) * batch_size * seq_len
        tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
        print(f"Step {step:3d} | Loss: {loss.item():.4f} | Tokens/sec: {tokens_per_sec:.1f}")

print(f"\n✅ NanoChat training successful on Windows CPU!")
print(f"Total time: {time.time() - start_time:.1f}s")