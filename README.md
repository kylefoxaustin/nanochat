# nanochat

![nanochat logo](dev/nanochat.png)

# nanochat - RTX 5090 Single GPU Fork + Windows 11 CPU Support

This fork adds optimized configurations and documentation for training nanochat on single GPU systems (specifically tested on NVIDIA RTX 5090 and RTX 8000) and full Windows 11 CPU-only training support.

## Fork Additions

### 🚀 Single GPU Training Scripts
- `speedrun_rtx5090.sh` - Optimized training pipeline for RTX 5090 (depth=16, 370M params)
- `speedrun_rtx8000.sh` - FP32 training pipeline for RTX 8000 (depth=12, 185M params)
- `train_big_model.sh` - Extended training for 740M parameter model (depth=24)

### 💻 Windows 11 CPU Training Scripts
- `train_340M_menu.py` - Interactive menu-driven training for 304M parameter model
- `train_340M_full.py` - Full 304M model training
- `train_340M_ultimate.py` - Extended configuration
- `train_cpu_simple_final.py` - Simplified CPU training
- `train_full_nanochat_cpu.py` - Standard nanochat CPU version
- `train_full_nanochat_final.py` - Final nanochat implementation
- `train_full_real_data.py` - Training with real datasets
- `train_full_real_data_fixed.py` - Fixed real data training
- `train_nanochat_cpu.py` - Basic CPU training
- `train_nanochat_final.py` - Final standard training
- `train_real_data_final.py` - Production data training
- `fix_final_tokenizer.py` - Tokenizer utilities

### 📊 Performance Benchmarks

#### GPU Performance
- **RTX 5090 Depth=16 (370M)**: ~80k tokens/sec, completes in ~30 hours
- **RTX 5090 Depth=24 (740M)**: ~57k tokens/sec, ~50 hours with proper iterations
- **RTX 8000 Depth=12 (185M)**: ~1.3k tokens/sec, ~60-70 hours (FP32 mode)

#### CPU Performance (Windows 11)
- **64M Model**: ~770 tokens/sec, ~4 hours for 10k iterations
- **304M Model**: ~64 tokens/sec, ~52-58 hours for 100k iterations
- Memory usage: 2-3GB RAM (64M), 5-6GB RAM (304M)
- No VRAM required

### 🛠️ Key Modifications
- Removed muon optimizer requirements for single GPU compatibility
- Added Windows 11 CPU-only training support with no CUDA dependencies
- Automatic GPT-2 tokenizer fallback for Windows compatibility
- Interactive menu system for easy training configuration
- Adjusted batch sizes for VRAM/RAM optimization
- Added extended training iterations for better model quality
- RTX 8000 uses proven learning rates to prevent NaN issues in FP32

### 📈 Training Results
- Successfully trained 370M model achieving 0.11 CORE score
- RL fine-tuning completed for improved response quality
- Currently training 740M model with 20k iterations

---

## 🪟 Windows 11 CPU Training

### Prerequisites
- Windows 11
- Python 3.11 or later
- Git
- 8GB+ RAM (16GB+ recommended)

### Windows Setup

1. **Clone the repository**
```powershell
git clone https://github.com/kylefoxaustin/nanochat.git
cd nanochat
git checkout windows-11-cpu-support
```

2. **Create virtual environment**
```powershell
python -m venv venv
venv\Scripts\activate
```

3. **Install PyTorch CPU version**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. **Install other dependencies**
```powershell
pip install transformers datasets wandb tqdm numpy pandas pyarrow
```

### Windows Training Options

#### Interactive Menu System (Recommended)
```powershell
python train_340M_menu.py
```
Offers 5 training configurations:
1. Test Run (100 iterations) - ~3 minutes
2. Quick Training (1,000 iterations) - ~30 minutes
3. Standard Training (10,000 iterations) - ~5 hours
4. Full Training (100,000 iterations) - ~52 hours
5. Custom (specify your own parameters)

#### Direct Script Execution
```powershell
# For 304M model
python train_340M_full.py

# For 64M model
python train_full_nanochat_cpu.py

# For real data training
python train_real_data_final.py
```

### Windows Checkpointing
Checkpoints automatically save to:
```
C:\Users\[username]\.cache\nanochat\340M_checkpoints\
```

Files saved:
- `checkpoint_iter_[N].pt` - Model weights
- `optimizer_iter_[N].pt` - Optimizer state
- `training_state_iter_[N].json` - Training metadata

### Windows-Specific Notes
- Scripts automatically fall back to GPT-2 tokenizer if default fails
- All paths use Windows-compatible formatting
- SSL certificate issues can be resolved with:
  ```powershell
  pip config set global.trusted-host "pypi.org files.pythonhosted.org download.pytorch.org"
  ```

---

## 🚀 GPU Quick Start

### Training on RTX 5090

#### 370M Parameter Model (depth=16)
```bash
# Clone this fork
git clone https://github.com/kylefoxaustin/nanochat.git
cd nanochat

# Run the optimized RTX 5090 training pipeline
chmod +x speedrun_rtx5090.sh
./speedrun_rtx5090.sh
```
**Expected runtime**: ~30 hours total
**VRAM usage**: ~18GB

#### 740M Parameter Model (depth=24)
- Extended training for 740M parameter model (depth=24)
- Complete 4-phase pipeline: Base → Midtraining → SFT → **RL**
- 20,000 iterations for proper convergence
- Includes reinforcement learning for improved response quality
- Produces a model approaching GPT-2 quality
- Note:  I ran this on RTX8000 and the estimated time to complete was 17 Days...

### Usage:

```bash
# Ensure you have 150 data shards first
python -m nanochat.dataset -n 150

# Run the extended training script
chmod +x train_big_model.sh
./train_big_model.sh
```

**Expected runtime:** ~55-60 hours total (Base: ~50h, Mid: ~2h, SFT: ~2h, RL: ~3h)
**VRAM usage:** ~25GB

### Training Phases:

1. **Base Training** (20,000 steps): Teaches the model language fundamentals
2. **Midtraining**: Adds instruction-following capabilities
3. **SFT** (Supervised Fine-Tuning): Optimizes for conversational interactions
4. **RL** (Reinforcement Learning): Reinforces high-quality responses and reasoning

The RL phase typically improves:
- Mathematical reasoning (GSM8K scores)
- Complex instruction following
- Overall response quality and coherence

### Training on RTX 8000

#### 185M Parameter Model (depth=12)
```bash
# Clone this fork
git clone https://github.com/kylefoxaustin/nanochat.git
cd nanochat

# Run the RTX 8000 FP32 training pipeline
chmod +x speedrun_rtx8000.sh
./speedrun_rtx8000.sh
```
**Expected runtime**: ~60-70 hours total
**VRAM usage**: ~20GB
**Note**: Uses FP32 precision with conservative learning rates to prevent NaN issues

### Hardware Requirements
- **For RTX 5090**: NVIDIA RTX 5090 (32GB VRAM), 96GB+ system RAM
- **For RTX 8000**: NVIDIA RTX 8000 (48GB VRAM), 64GB+ system RAM
- ~50GB storage for data shards
- Ubuntu 22.04+ with CUDA 13.0+

### What These Scripts Do
1. **speedrun_rtx5090.sh**: Modified version of original speedrun.sh
   - Optimized batch sizes for single GPU
   - Removed distributed training dependencies
   - Complete pipeline: base training → midtraining → SFT

2. **speedrun_rtx8000.sh**: FP32-specific implementation
   - Conservative learning rates (matrix_lr=0.005, embedding_lr=0.001)
   - Gradient clipping for stability
   - Designed for older GPU architectures without BF16 support

3. **train_big_model.sh**: For ambitious single-GPU training
   - 20,000 iterations for proper convergence
   - Larger batch accumulation for stability
   - Produces a model approaching GPT-2 quality

---

## 🔧 Troubleshooting

### Common GPU Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in the scripts
--device_batch_size=2  # or even 1 if needed
```

**2. "Unknown config key: use_muon" Error**
- Already fixed in these scripts by removing the --muon flag
- If modifying, avoid bare flags without values

**3. FileNotFoundError for checkpoints**
- This happens if continuing from a previous run
- Clear old checkpoints: `rm -rf ~/.cache/nanochat/*_checkpoints`

**4. NaN Loss Issues**
- The scripts use conservative learning rates to prevent this
- If it occurs, check that you're using the exact parameters in the scripts

**5. Python/Module Not Found**
```bash
# Always activate the virtual environment first
source .venv/bin/activate
# Use python3 if python command not found
alias python=python3
```

**6. Slow Training / Low GPU Utilization**
```bash
# Check GPU status
nvidia-smi
# Ensure no other processes are using GPU
# Temperature should be 60-70°C under load
```

**7. Data Download Issues**
```bash
# If data download is interrupted
python -m nanochat.dataset -n 150  # Will skip existing files
```

### Common Windows CPU Issues

**1. Import Errors**
```powershell
# Ensure virtual environment is activated
venv\Scripts\activate
```

**2. Memory Errors**
```python
# Reduce batch size in training script
batch_size = 4  # Reduce from 8
```

**3. Tokenizer Errors**
- Scripts automatically fall back to GPT-2 tokenizer
- No manual intervention needed

**4. Line Ending Warnings in Git**
- Normal on Windows when committing
- Git handles CRLF/LF conversion automatically
- Safe to ignore

---

## 📊 Expected Metrics During Training
- **Starting loss**: ~11.0
- **Good progress**: Loss dropping by ~0.1 every 100 steps
- **Healthy GPU**: 90%+ utilization, 60-70°C
- **Token rate**: 80k+ for depth=16, 57k+ for depth=24

## 📈 Monitoring Training Progress

### Real-time Monitoring
```bash
# Linux/GPU
tail -f screenlog.0  # if using screen

# Windows/CPU - watch terminal directly or:
# Training outputs to console and training_log.txt
```

### Key Metrics to Watch
- **Loss**: Should steadily decrease from ~11.0 to target
- **Tokens/sec**: Consistent rate indicates healthy training
- **MFU (Model FLOPs Utilization)**: 15-17 is excellent (GPU only)
- **Validation bpb**: Lower is better (bits per byte)
- **Progress %**: Shows completion status
- **ETA**: Estimated time remaining

### Check GPU Status
```bash
# Monitor GPU usage and temperature
watch -n 2 nvidia-smi

# Healthy indicators:
# - GPU Utilization: 90%+
# - Memory Usage: 18-25GB (depending on model)
# - Temperature: 60-70°C
# - Power: 350-450W
```

### Training Checkpoints
```bash
# View saved checkpoints
ls ~/.cache/nanochat/base_checkpoints/
ls ~/.cache/nanochat/chatsft_checkpoints/

# Backup important checkpoints
cp -r ~/.cache/nanochat/*_checkpoints /mnt/backup/
```

### Estimate Completion Time
- **Depth=16**: ~800ms/step × 102,400 steps ÷ 3600 = ~23 hours
- **Depth=24**: ~9s/step × 20,000 steps ÷ 3600 = ~50 hours

### If Training Crashes
```bash
# Models auto-save every 300 steps
# To resume, check the latest checkpoint:
ls ~/.cache/nanochat/base_checkpoints/*/model_*.pt
# Modify training script to resume from checkpoint (advanced)
```

### Testing Your Model
```bash
# Quick test after training
python -m scripts.chat_cli -p "Hello, how are you?"

# Full web interface
python -m scripts.chat_web
# Visit http://localhost:8000
```

---

## 🤝 Contributing

This fork is maintained by [@kylefoxaustin](https://github.com/kylefoxaustin). Contributions are welcome!

If you:
- Have improvements for single-GPU or CPU training
- Find optimizations for other hardware configurations
- Want to share your training results
- Have bug fixes or documentation improvements

Please feel free to:
1. Open an issue to discuss your idea
2. Submit a pull request with your changes
3. Share your benchmarks in the discussions

## 📊 Community Benchmarks

Have you successfully trained on different hardware? Please share your results!

| Hardware | Model Size | Tokens/sec | Training Time | Memory Usage | Platform | Contributor |
|----------|------------|------------|---------------|--------------|----------|-------------|
| RTX 5090 | 370M (d16) | 80k | 30 hours | 18GB VRAM | Linux | @kylefoxaustin |
| RTX 5090 | 740M (d24) | 57k | 50 hours | 25GB VRAM | Linux | @kylefoxaustin |
| RTX 8000 | 185M (d12) | 1.3k | 60-70 hours | 20GB VRAM | Linux | @kylefoxaustin |
| CPU (Modern) | 64M | 770 | 4 hours (10k) | 3GB RAM | Windows 11 | @kylefoxaustin |
| CPU (Modern) | 304M | 64 | 52 hours (100k) | 6GB RAM | Windows 11 | @kylefoxaustin |
| *Your Hardware* | *Your results* | *PR welcome!* | | | | |

---

From the original karpathy/nanochat repository's README
> The best ChatGPT that $100 can buy.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## Talk to it

To get a sense of the endpoint of this repo, you can currently find [nanochat d32](https://github.com/karpathy/nanochat/discussions/8) hosted on [nanochat.karpathy.ai](https://nanochat.karpathy.ai/). "d32" means that this model has 32 layers in the Transformer neural network. This model has 1.9 billion parameters, it was trained on 38 billion tokens by simply running the single script [run1000.sh](run1000.sh), and the total cost of training was ~$800 (about 33 hours training time on 8XH100 GPU node). While today this is enough to outperform GPT-2 of 2019, it falls dramatically short of modern Large Language Models like GPT-5. When talking to these micro models, you'll see that they make a lot of mistakes, they are a little bit naive and silly and they hallucinate a ton, a bit like children. It's kind of amusing. But what makes nanochat unique is that it is fully yours - fully configurable, tweakable, hackable, and trained by you from start to end. To train and talk to your own, we turn to...

## Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensate by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.

## Running on CPU / MPS

nanochat can be run on CPU or on MPS (if you're on Macbook), and will automatically try to detect what device is best to run on. You're not going to get too far without GPUs, but at least you'll be able to run the code paths and maybe train a tiny LLM with some patience. For an example of how to make all the run commands much smaller (feel free to tune!), you can refer to [dev/runcpu.sh](dev/runcpu.sh) file. You'll see that I'm essentially restricting all scripts to train smaller models, to run for shorter number of iterations, etc. This functionality is new, slightly gnarly (touched a lot of code), and was merged in this [CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88) on Oct 21, 2025.

## Customization

To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into midtraining and SFT stages.

Additionally, to add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).

## Questions

nanochat is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

This includes all py, rs, html, toml, sh files, excludes the `rustbpe/target` folder, and chooses the cxml output format. Everything is written to the `packaged.txt` file, which atm measures ~330KB (i.e. well below ~100K tokens for a state of the art LLM), and ~8K lines of code in 45 files.

Alternatively, I recommend using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.

## Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # Pretraining data shard generation
│   └── runcpu.sh                   # Small example of how to run on CPU/MPS
├── nanochat
│   ├── __init__.py                 # empty
│   ├── adamw.py                    # Distributed AdamW optimizer
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── configurator.py             # A superior alternative to argparse
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── muon.py                     # Distributed Muon optimizer
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── run1000.sh                      # Train the ~$800 nanochat d32
├── rustbpe                         # Custom Rust BPE tokenizer trainer
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md                   # see for why this even exists
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py                # Base model: calculate CORE score
│   ├── base_loss.py                # Base model: calculate bits per byte, sample
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model (SFT/Mid): talk to over CLI
│   ├── chat_eval.py                # Chat model (SFT/Mid): eval tasks
│   ├── chat_rl.py                  # Chat model (SFT/Mid): reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model (SFT/Mid): talk to over WebUI
│   ├── mid_train.py                # Chat model: midtraining
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── speedrun.sh                     # Train the ~$100 nanochat d20
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_rustbpe.py
└── uv.lock
```

## Contributing

nanochat is nowhere near finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

Current LLM policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Windows 11 CPU support and single-GPU optimizations by [@kylefoxaustin](https://github.com/kylefoxaustin)

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT