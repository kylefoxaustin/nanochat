#!/usr/bin/env python
"""
Windows-compatible training script for nanochat
Supports both GPU (RTX 5090/8000) and CPU-only training
"""

import os
import sys
import platform
import subprocess
import argparse
import torch
from pathlib import Path

class NanoChatWindowsTrainer:
    def __init__(self, device_type='auto'):
        self.system = platform.system()
        self.device_type = self._setup_device(device_type)
        self.base_dir = Path.home() / '.cache' / 'nanochat'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"""

     NanoChat Windows Training System                 
     System: {self.system:41} 
     Device: {self.device_type:41} 
     Python: {sys.version.split()[0]:41} 
     PyTorch: {torch.__version__:40} 

        """)
        
    def _setup_device(self, device_type):
        if device_type == 'auto':
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f" GPU detected: {gpu_name}")
                return 'cuda'
            else:
                print(" Running in CPU mode")
                return 'cpu'
        return device_type
    
    def test_setup(self):
        """Test the environment setup"""
        print("\nTesting environment...")
        
        # Test PyTorch
        device = 'cuda' if 'cuda' in self.device_type else 'cpu'
        test_tensor = torch.randn(100, 100).to(device)
        result = test_tensor @ test_tensor.T
        print(f" PyTorch working on {device}")
        
        # Check for nanochat modules
        if not Path("nanochat").exists():
            print(" Warning: nanochat/ folder not found")
            return False
        
        if not Path("scripts").exists():
            print(" Warning: scripts/ folder not found")
            return False
            
        print(" Core folders present")
        print(" All checks passed!")
        return True

def main():
    parser = argparse.ArgumentParser(description='NanoChat Windows Trainer')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], 
                       default='auto', help='Device to use')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test only')
    
    args = parser.parse_args()
    
    trainer = NanoChatWindowsTrainer(device_type=args.device)
    
    if args.quick:
        if trainer.test_setup():
            print("\n Environment ready for training!")
            print("\nNext steps:")
            print("1. Download data: python -m nanochat.dataset -n 5")
            print("2. Train tokenizer: python -m scripts.tok_train")
            print("3. Start training: python -m scripts.base_train")

if __name__ == "__main__":
    main()
