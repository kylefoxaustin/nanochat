import tiktoken
import pickle
from pathlib import Path

print("Using GPT-2 tokenizer as substitute...")
enc = tiktoken.get_encoding("gpt2")
print(f"Vocab size: {enc.n_vocab}")

tok_path = Path.home() / ".cache" / "nanochat" / "tokenizer.pkl"
tok_path.parent.mkdir(exist_ok=True)

with open(tok_path, 'wb') as f:
    pickle.dump(enc, f)

print(f"✓ Saved tokenizer to {tok_path}")
print("Ready to continue with training!")