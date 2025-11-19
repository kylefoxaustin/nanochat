import pickle
import tiktoken
from pathlib import Path

# Create a tokenizer that nanochat's RustBPETokenizer can work with
class NanoChatTokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = 50257
        
    def encode_single_token(self, text):
        # Handle special tokens that nanochat expects
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        special = {
            '<|bos|>': 0,
            '<|eos|>': 1, 
            '<|pad|>': 2,
        }
        if text in special:
            return special[text]
        try:
            return self.enc.encode_single_token(text)
        except:
            return 0  # fallback
    
    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, tokens):
        return self.enc.decode(tokens)

# Save it where nanochat expects
tokenizer_path = Path.home() / '.cache' / 'nanochat' / 'tokenizer' / 'tokenizer.pkl'
tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

tokenizer = NanoChatTokenizer()
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# Also save the class definition so pickle can find it
import sys
sys.modules['__main__'].NanoChatTokenizer = NanoChatTokenizer

print(f"✓ NanoChat-compatible tokenizer saved to {tokenizer_path}")