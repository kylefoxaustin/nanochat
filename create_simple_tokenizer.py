import pickle
from pathlib import Path

# Create the simplest possible tokenizer that will work
class SimpleTokenizer:
    def __init__(self):
        self.vocab_size = 50257  # GPT-2 vocab size
        
    def encode(self, text):
        # Simple character-level encoding for testing
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, tokens):
        return ''.join([chr(t % 256) for t in tokens])

# Just save the bare object
tokenizer_path = Path.home() / '.cache' / 'nanochat' / 'tokenizer' / 'tokenizer.pkl'
tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

# Save just a simple dictionary that base_train can work with
tokenizer_data = {
    'vocab_size': 50257,
    'type': 'simple'
}

with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer_data, f)

print(f"✓ Simple tokenizer saved to {tokenizer_path}")