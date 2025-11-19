import torch
import torch.nn as nn
import time

print("Simple PyTorch CPU Training Test")
print("="*50)

# Fixed model - handle LSTM output properly
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.output = nn.Linear(256, 1000)
    
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)  # LSTM returns (output, (hidden, cell))
        x = self.output(x)
        return x

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters())
print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
print("\nStarting training on CPU...")

# Train
start = time.time()
for step in range(50):
    x = torch.randint(0, 1000, (4, 32))
    y = torch.randint(0, 1000, (4, 32))
    
    out = model(x)
    loss = nn.functional.cross_entropy(out.reshape(-1, 1000), y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        elapsed = time.time() - start
        tokens_per_sec = (step + 1) * 4 * 32 / elapsed if elapsed > 0 else 0
        print(f"Step {step:3d} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s | Tokens/sec: {tokens_per_sec:.1f}")

print(f"\n✓ Training successful! Total time: {time.time()-start:.1f}s")
print("Your Windows 11 CPU can train neural networks!")