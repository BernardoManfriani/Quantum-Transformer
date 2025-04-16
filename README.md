# Quantum-Transformer

This project explores a hybrid quantum-classical architecture for sequence generation based on a Transformer model. Inspired by the work of Smaldone et al. ([arXiv:2502.19214](https://arxiv.org/pdf/2502.19214)), it integrates Parametrized Quantum Circuits (PQC) within the self-attention mechanism to enhance expressivity in molecular and text generation tasks. The model supports both Dante-style text generation and molecular string generation in SMILES format.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Model Training](#model-training)
  - [Dante Training](#dante-training)
  - [Quick Training (Debug/Test)](#quick-training-debugtest)
  - [Custom Training Options](#custom-training-options)
- [Text and SMILES Generation](#text-and-smiles-generation)
  - [Dante Text Generation](#dante-text-generation)
  - [SMILES Generation](#smiles-generation)
- [Usage Examples](#usage-examples)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [References](#references)

---

## Features
- **Hybrid Quantum Transformer**: Incorporates quantum variational circuits into the attention mechanism (QK^T calculation via Hadamard test).
- **Dual-Task Model**: Supports generation of Dante-style poetry and SMILES strings.
- **Attention via Quantum Circuits**: Implements query-key similarity with PQC and real part of inner product measurement.
- **Efficient Gradient Optimization**: Supports Parameter-Shift Rule and SPSA for quantum parameter training.
- **QPU Parallelization**: Distributes circuit execution with `cudaq.observe_async()` across multiple QPUs.
- **Character-level Tokenizer**: For Dante domain adaptation.
- **Flexible CLI Interface**: Easy configuration for training and generation.

---

## Requirements
```bash
Python >= 3.8
pip install cudaq rdkit==2024.9.4
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torchvision torchaudio pandas==2.2.2 torchdata==0.10.1
pip install tqdm==4.67.1 scikit-learn==1.5.1 seaborn==0.13.2 gdown==5.2.0
```

---

## Project Structure
```plaintext
Quantum-Transformer/
├── main.py                  # Entry point for training/generation
├── run_dante.py             # Script for training Dante model
├── reproduce.py             # Placeholder
├── dataset/
│   └── inferno.txt          # Dante's Inferno text (included)
├── model_checkpoints/
│   ├── dante/               # Dante model checkpoints
│   └── smile/               # SMILES model checkpoints
├── src/
│   ├── dante_trainer.py     # Dante training logic
│   ├── quantum_layer.py     # Custom quantum layers (attention)
│   ├── quantum_transformer.py # Model architecture with quantum attention
│   ├── utils.py             # Tokenization, generation, sampling
│   └── unitary_library.py   # CUDA-Q kernels for circuits
└── README.md                # This file
```

---

## Datasets
- **Text**: `dataset/inferno.txt` (already included)
- **SMILES**: Not included; must be user-provided

---

## Model Training

### Dante Training
```bash
python main.py trainDante
# or:
python run_dante.py
```

#### Key Options:
```bash
--epochs N         # Number of epochs (default: 20)
--batch_size N     # Batch size (default: 16)
--block_size N     # Sequence length (default: 128)
--lr VAL           # Learning rate (default: 3e-4)
--save_every N     # Save checkpoint every N epochs (default: 5)
--fast             # Enable fast/debug mode
```

#### Example:
```bash
python main.py trainDante --epochs 50 --batch_size 32 --lr 5e-4
```

### Quick Training (Debug/Test)
```bash
python main.py trainDante --fast
```

---

## Text and SMILES Generation

### Dante Text Generation
```bash
python main.py generate --model dante --prompt "Nel mezzo del cammin"
```
#### Additional Options:
```bash
--max_len N        # Max length of generated sequence
--temperature VAL  # Sampling temperature
--top_k N          # Top-k sampling
```
#### Example:
```bash
python main.py generate --model dante --prompt "Nel mezzo del cammin" --max_len 200 --temperature 0.7 --top_k 40
```

### SMILES Generation
```bash
python main.py generate --model smile --prompt "C"
```
#### Custom Example:
```bash
python main.py generate --model smile --prompt "C" --max_len 50 --temperature 0.8 --top_k 5
```

---

## Usage Examples

### From Python Code
```python
from src.quantum_transformer import QuantumTransformerModel
from src.utils import generate_text
import torch

model = QuantumTransformerModel(
    qpu_count=1,
    vocab_size=78,
    embed_dim=64,
    block_size=128,
    num_qubits=6,
    ansatz_layers=1
)
checkpoint = torch.load('./model_checkpoints/dante/best_dante_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

generated = generate_text(
    model=model,
    prompt_tokens=['[CLS]'] + list("Nel mezzo del cammin"),
    max_len=100,
    temperature=0.7,
    top_k=40
)
print(generated)
```

---

## Tips & Troubleshooting
- **Quantum Gradient Optimization**: Choose between Parameter-Shift and SPSA (recommended for speed).
- **Ancilla Register**: Ensure correct ancilla initialization for Hadamard tests.
- **SMILES Preprocessing**: Tokenization is regex-based with canonicalization via RDKit.
- **Character-Level Tokenizer**: Used for Dante generation; SMILES uses chemistry-specific tokens.
- **Multi-QPU Support**: Enabled via `cudaq.observe_async`, efficient for large attention matrices.

---

## References
- [Smaldone et al. (2025)](https://arxiv.org/pdf/2502.19214)
- [cuda-quantum (NVIDIA)](https://github.com/NVIDIA/cuda-quantum)
- [PyTorch](https://pytorch.org/)
- [RDKit](https://www.rdkit.org/)
- Dante Alighieri, "La Divina Commedia" (Project Gutenberg)

---

For questions, bugs, or contributions, please open a GitHub issue or contact the maintainers.
