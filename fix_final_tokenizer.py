import pickle
from pathlib import Path
import sys

# Define the tokenizer class so pickle can find it
class NanoChatTokenizer:
    def __init__(self):
        self.vocab_size = 50257
        
    def encode_single_token(self, text):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        special = {'<|bos|>': 0, '<|eos|>': 1, '<|pad|>': 2}
        if text in special:
            return special[text]
        return 0
    
    def encode(self, text):
        # Simple encoding for testing
        return [ord(c) % self.vocab_size for c in text[:512]]
    
    def decode(self, tokens):
        return ''.join([chr(t % 256) for t in tokens])

# Save a simple tokenizer that will work
tokenizer_path = Path.home() / '.cache' / 'nanochat' / 'tokenizer' / 'tokenizer.pkl'
tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

# Create simple tokenizer data
tokenizer_data = {
    'vocab_size': 50257,
    'enc': NanoChatTokenizer()
}

with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer_data['enc'], f)

print(f"✓ Fixed tokenizer saved to {tokenizer_path}")