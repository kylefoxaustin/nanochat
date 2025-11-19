import pickle
from pathlib import Path
import tiktoken

# Create a wrapper that handles special tokens
class TokenizerWrapper:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
        
    def encode_single_token(self, text):
        # Map special tokens to IDs
        special_tokens = {
            b'<|bos|>': 0,
            b'<|eos|>': 1,
            b'<|pad|>': 2,
        }
        if text in special_tokens:
            return special_tokens[text]
        return self.enc.encode_single_token(text)
    
    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, tokens):
        return self.enc.decode(tokens)

# Save the wrapper
tokenizer_path = Path.home() / '.cache' / 'nanochat' / 'tokenizer' / 'tokenizer.pkl'
tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

wrapper = TokenizerWrapper()
with open(tokenizer_path, 'wb') as f:
    pickle.dump(wrapper, f)

print(f"✓ Fixed tokenizer saved to {tokenizer_path}")