import os
import sys
import torch
import time
from pathlib import Path
import psutil

def show_header():
    print("\n" + "="*60)
    print("""
    304M Parameter NanoChat - CPU Training Suite
           Windows 11 Optimized Edition
    """)
    print("="*60)

def show_system_info():
    print(f"\nSystem Information:")
    print(f"  CPU: {psutil.cpu_count()} cores ({psutil.cpu_count(logical=False)} physical)")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

def main_menu():
    while True:
        print("\n" + "="*60)
        print("TRAINING MENU")
        print("="*60)
        print("1. Quick Test       - 1,000 iterations   (~1 hour)")
        print("2. Proof of Concept - 10,000 iterations  (~10 hours)")  
        print("3. Serious Training - 50,000 iterations  (~50 hours / 2 days)")
        print("4. FULL NANOCHAT    - 100,000 iterations (~100 hours / 4 days)")
        print("5. Custom           - Choose your own number")
        print("6. Quit")
        print("="*60)
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '6':
            print("\nExiting training suite. Goodbye!")
            sys.exit(0)
        
        iteration_map = {
            '1': 1000,
            '2': 10000,
            '3': 50000,
            '4': 100000,
        }
        
        if choice == '5':
            try:
                num_iterations = int(input("Enter number of iterations: "))
            except ValueError:
                print("Invalid number. Please try again.")
                continue
        elif choice in iteration_map:
            num_iterations = iteration_map[choice]
        else:
            print("Invalid choice. Please select 1-6.")
            continue
        
        # Calculate time estimate (based on your ~60 tokens/sec benchmark)
        estimated_hours = num_iterations * 3.5 / 3600  # 3.5 seconds per iteration from your benchmark
        estimated_days = estimated_hours / 24
        
        # Show confirmation
        print("\n" + "-"*60)
        print(f"TRAINING CONFIGURATION:")
        print(f"  Model size: 304M parameters")
        print(f"  Iterations: {num_iterations:,}")
        print(f"  Estimated time: {estimated_hours:.1f} hours", end="")
        if estimated_days > 1:
            print(f" ({estimated_days:.1f} days)")
        else:
            print()
        print(f"  Checkpoint saves: Every {min(500, num_iterations // 10)} iterations")
        print("-"*60)
        
        confirm = input("\nStart training? (y/n): ").strip().lower()
        
        if confirm == 'y':
            run_training(num_iterations)
            break  # Exit after training completes
        else:
            print("Training cancelled. Returning to menu.")
            continue

def run_training(num_iterations):
    print("\n🚀 Initializing training...")
    
    sys.path.append('.')
    from nanochat.gpt import GPT, GPTConfig
    
    # CPU optimizations
    num_threads = psutil.cpu_count(logical=False)
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    
    # Model configuration
    config = GPTConfig(
        sequence_len=1024,
        vocab_size=50257,
        n_layer=16,
        n_head=16,
        n_kv_head=16,
        n_embd=1024,
    )
    
    print("Creating model...")
    model = GPT(config).to('cpu')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params/1e6:.1f}M parameters")
    
    # Training setup
    batch_size = 1
    seq_length = 128
    learning_rate = 3e-4
    save_every = min(500, num_iterations // 10)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    print("✓ Optimizer ready")
    
    # Checkpoint directory
    checkpoint_dir = Path.home() / '.cache' / 'nanochat' / '340M_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoints
    existing_checkpoints = list(checkpoint_dir.glob('model_340M_step_*.pt'))
    start_iteration = 0
    
    if existing_checkpoints:
        print(f"\n⚠️  Found {len(existing_checkpoints)} existing checkpoints")
        resume = input("Resume from latest checkpoint? (y/n): ").strip().lower()
        
        if resume == 'y':
            latest_checkpoint = max(existing_checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iteration = checkpoint['iteration'] + 1
            print(f"✓ Resumed from iteration {start_iteration}")
    
    print(f"\n" + "="*60)
    print(f"STARTING TRAINING")
    print(f"  Iterations: {start_iteration} → {num_iterations}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("  Press Ctrl+C to safely stop anytime")
    print("="*60 + "\n")
    
    # Training loop
    start_time = time.time()
    model.train()
    
    try:
        for i in range(start_iteration, num_iterations):
            # Generate batch
            x = torch.randint(0, 50257, (batch_size, seq_length))
            y = torch.roll(x, -1, 1)
            
            # Learning rate schedule
            lr_mult = 0.5 * (1 + torch.cos(torch.tensor(3.14159 * i / num_iterations)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_mult
            
            # Forward/backward
            loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Logging
            if i % 10 == 0:
                elapsed = time.time() - start_time
                iter_done = i - start_iteration + 1
                tokens_per_sec = iter_done * batch_size * seq_length / elapsed
                eta_hours = (num_iterations - i) * elapsed / iter_done / 3600 if iter_done > 0 else 0
                progress = 100 * i / num_iterations
                
                print(f"Iter {i:6d}/{num_iterations} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Tokens/s: {tokens_per_sec:.1f} | "
                      f"Progress: {progress:.1f}% | "
                      f"ETA: {eta_hours:.1f}h")
            
            # Save checkpoint
            if i % save_every == 0 and i > 0:
                checkpoint_path = checkpoint_dir / f'model_340M_step_{i}.pt'
                torch.save({
                    'iteration': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': config,
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved: step {i}")
                
                # Keep only last 5 checkpoints
                all_checkpoints = sorted(checkpoint_dir.glob('model_340M_step_*.pt'))
                if len(all_checkpoints) > 5:
                    for old_checkpoint in all_checkpoints[:-5]:
                        old_checkpoint.unlink()
                        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted! Saving checkpoint...")
        i = i if 'i' in locals() else start_iteration
    
    # Final save
    final_checkpoint = checkpoint_dir / f'model_340M_final_{i}.pt'
    torch.save({
        'iteration': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if 'loss' in locals() else 0,
        'config': config,
    }, final_checkpoint)
    
    # Training summary
    elapsed_total = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"  Total iterations trained: {i - start_iteration}")
    print(f"  Total time: {elapsed_total/3600:.2f} hours")
    if i - start_iteration > 0:
        print(f"  Average tokens/sec: {(i - start_iteration) * batch_size * seq_length / elapsed_total:.1f}")
        print(f"  Final loss: {loss.item() if 'loss' in locals() else 'N/A'}")
    print(f"  Model saved to: {final_checkpoint}")
    print("="*60)
    print("\n🎉 Your 304M parameter NanoChat model training is complete!")

if __name__ == "__main__":
    show_header()
    show_system_info()
    main_menu()