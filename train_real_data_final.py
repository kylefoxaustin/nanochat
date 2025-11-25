import os
import sys
import torch
import time
from pathlib import Path
import pandas as pd
import random

print("""
╔══════════════════════════════════════════════════════╗
║     FULL NanoChat Training - Windows CPU Edition     ║
║              With REAL Training Data                 ║
╚══════════════════════════════════════════════════════╝
""")

sys.path.append('.')
from nanochat.gpt import GPT, GPTConfig

# CPU optimizations
torch.set_num_threads(8)
os.environ['OMP_NUM_THREADS'] = '8'

# Load real data
data_dir = Path.home() / '.cache' / 'nanochat' / 'base_data'
data_files = list(data_dir.glob('*.parquet'))
print(f"Found {len(data_files)} data files")

# Load text from parquet files
print("Loading real text data...")
all_text = []
for file in data_files[:2]:  # Start with 2 files for memory efficiency
    df = pd.read_parquet(file)
    if 'text' in df.columns:
        all_text.extend(df['text'].tolist())
print(f"Loaded {len(all_text)} text samples")

# Simple tokenization (character-level for simplicity)
def encode(text, max_len=512):
    return [ord(c) % 50257 for c in text[:max_len]]

def get_batch(batch_size=2, seq_len=256):
    batch_x = []
    for _ in range(batch_size):
        text = random.choice(all_text)
        tokens = encode(text, seq_len + 1)
        if len(tokens) < seq_len + 1:
            tokens = tokens + [0] * (seq_len + 1 - len(tokens))
        batch_x.append(tokens[:seq_len])
    
    x = torch.tensor(batch_x)
    y = torch.roll(x, -1, 1)
    return x, y

# Model configuration - match your request
config = GPTConfig(
    sequence_len=512,
    vocab_size=50257,
    n_layer=4,
    n_head=8,
    n_kv_head=8,
    n_embd=512,
)

model = GPT(config).to('cpu')
print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# Training
num_iters = 10000
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"\nTraining on REAL DATA for {num_iters} iterations...")
print("="*60 + "\n")

start_time = time.time()
model.train()

for i in range(num_iters):
    x, y = get_batch(batch_size=2, seq_len=256)
    
    loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if i % 10 == 0:
        elapsed = time.time() - start_time
        tokens_per_sec = (i + 1) * 2 * 256 / elapsed
        eta_hours = (num_iters - i) * elapsed / (i + 1) / 3600
        print(f"Iter {i:5d}/{num_iters} | Loss: {loss.item():.4f} | "
              f"Tokens/s: {tokens_per_sec:.0f} | ETA: {eta_hours:.1f}h")

print(f"\n✅ Training complete on REAL DATA!")