import torch
import pickle
from pathlib import Path

tok_path = Path.home() / '.cache' / 'nanochat' / 'tokenizer.pkl'
print(f'✓ Tokenizer exists: {tok_path.exists()}')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Device: {"CUDA" if torch.cuda.is_available() else "CPU"}')
print('\nReady for training!')