import sys
sys.path.append('.')
from nanochat.gpt import GPTConfig
import inspect

# Check what parameters GPTConfig accepts
sig = inspect.signature(GPTConfig.__init__)
print("GPTConfig parameters:")
for param in sig.parameters:
    print(f"  {param}")