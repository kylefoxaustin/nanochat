\# NanoChat Windows Edition



A Windows 11-compatible fork of \[karpathy/nanochat](https://github.com/karpathy/nanochat) with CPU training support and Windows-specific optimizations.



\## 🚀 Features



\- ✅ \*\*Full Windows 11 Compatibility\*\* - Native Windows support without WSL

\- ✅ \*\*CPU Training Support\*\* - Optimized for AMD64 processors

\- ✅ \*\*GPU Ready\*\* - Tested with RTX 5090 and RTX 8000

\- ✅ \*\*Tokenizer Workaround\*\* - GPT-2 tokenizer solution for Windows

\- ✅ \*\*Simple Setup\*\* - Standard Python venv, no complex dependencies



\## 📋 Requirements



\### System Requirements

\- Windows 10/11 (64-bit)

\- Python 3.12+

\- 16GB+ RAM (32GB recommended)

\- 50GB+ free disk space



\### For GPU Training (Optional)

\- NVIDIA RTX GPU with 8GB+ VRAM

\- CUDA 12.1+

\- Latest NVIDIA drivers



\## 🛠️ Installation



\### 1. Clone the Repository

```bash

git clone -b windows-11-cpu-support https://github.com/kylefoxaustin/nanochat.git

cd nanochat

```



\### 2. Set Up Python Environment

```powershell

\# Create virtual environment

python -m venv venv



\# Activate it (Windows PowerShell)

.\\venv\\Scripts\\Activate.ps1



\# If you get an execution policy error:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

```



\### 3. Install Dependencies

```powershell

\# Upgrade pip

python -m pip install --upgrade pip



\# Install PyTorch (CPU version)

pip install torch torchvision torchaudio



\# For GPU (CUDA 12.1)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121



\# Install other requirements

pip install numpy tqdm datasets transformers fastapi uvicorn pydantic pytest wandb certifi tiktoken

```



\### 4. Set Up Tokenizer

```powershell

\# Run the GPT-2 tokenizer setup (Windows workaround)

python use\_gpt2\_tokenizer.py

```



\### 5. Download Training Data

```powershell

\# Download training data (adjust number of shards as needed)

python -m nanochat.dataset -n 5

```



\## 🎯 Quick Start



\### Test Your Setup

```powershell

\# Verify environment

python test\_setup.py



\# Quick training test

python windows\_train.py --quick --device cpu

```



\### Start Full CPU Training

```powershell

\# Run full training pipeline on CPU

python windows\_train.py --device cpu



\# Or for smaller test run

python -m scripts.base\_train --depth=4 --device\_batch\_size=1 --max\_iters=100

```



\## 📁 Project Structure

```

nanochat-windows/

├── windows\_train.py        # Windows training orchestrator

├── use\_gpt2\_tokenizer.py   # Tokenizer workaround for Windows

├── test\_setup.py          # Environment verification script

├── nanochat/              # Core nanochat modules

├── scripts/               # Training scripts

├── tasks/                 # Training tasks

└── venv/                  # Virtual environment (not in repo)

```



\## ⚙️ Configuration



\### CPU Training Settings

For CPU training, use smaller models and batch sizes:

```python

\# Recommended CPU settings (in windows\_train.py)

--depth=4                  # Smaller model depth

--device\_batch\_size=1      # Minimal batch size

--batch\_size=2            # Total batch size

--max\_iters=1000          # Fewer iterations

--device=cpu              # Force CPU mode

```



\### GPU Training Settings

If you have an NVIDIA GPU:

```python

\# RTX 5090 settings

--depth=16

--device\_batch\_size=32

--max\_iters=102400



\# RTX 8000 settings (FP32 mode)

--depth=12

--device\_batch\_size=16

--precision=float32

```



\## 🐛 Troubleshooting



\### SSL Certificate Errors

```powershell

python -c "import certifi; import os; os.environ\['REQUESTS\_CA\_BUNDLE'] = certifi.where()"

```



\### PowerShell Execution Policy

```powershell

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

```



\### Missing Modules

```powershell

pip install \[missing\_module\_name]

```



\## 📊 Performance Expectations



\### CPU Training

\- \*\*Time\*\*: 48-72 hours for small model

\- \*\*RAM Usage\*\*: 8-16GB

\- \*\*Model Size\*\*: 100M parameters recommended



\### GPU Training

\- \*\*RTX 5090\*\*: ~30 hours for 370M params

\- \*\*RTX 8000\*\*: ~60 hours for 185M params (FP32)



\## 🤝 Contributing



Contributions are welcome! Please:

1\. Fork the repository

2\. Create a feature branch

3\. Test on Windows 11

4\. Submit a pull request



\## 📜 License



MIT License - Same as original nanochat



Copyright (c) 2024 Kyle Fox



Permission is hereby granted, free of charge, to any person obtaining a copy

of this software and associated documentation files (the "Software"), to deal

in the Software without restriction, including without limitation the rights

to use, copy, modify, merge, publish, distribute, sublicense, and/or sell

copies of the Software, and to permit persons to whom the Software is

furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in all

copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,

FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE

AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER

LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

SOFTWARE.



\## 👨‍💻 Maintainer



\*\*Kyle Fox\*\* - \[@kylefoxaustin](https://github.com/kylefoxaustin)

\- Windows compatibility implementation

\- CPU training optimization

\- Tokenizer workaround solution



\## 🙏 Acknowledgments



\- Original \[nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy

\- Windows testing on RTX 5090 and RTX 8000 systems

\- GPT-2 tokenizer from OpenAI via tiktoken



\## 📞 Support



\- \*\*Issues\*\*: \[GitHub Issues](https://github.com/kylefoxaustin/nanochat/issues)

\- \*\*Discussions\*\*: \[GitHub Discussions](https://github.com/kylefoxaustin/nanochat/discussions)



---



\*Last updated: November 2024\*

\*Tested on: Windows 11, Python 3.12.10, PyTorch 2.9.1+cpu\*

