# Quantum-Transformer

Questo progetto nasce per la generazione di testo e sequenze molecolari (SMILES) tramite un modello ispirato ai Transformer proposto da Smaldone et al. ["A Hybrid Transformer Architecture with a Quantized Self-Attention Mechanism Applied to Molecular Generatione"](https://arxiv.org/pdf/2502.19214) arricchito da layer quantistici custom. Il sistema supporta sia la generazione di testo in stile dantesco (Inferno di Dante) sia la generazione di stringhe SMILES per la chimica computazionale.

---

## Indice
- [Caratteristiche](#caratteristiche)
- [Requisiti](#requisiti)
- [Struttura del progetto](#struttura-del-progetto)
- [Dataset](#dataset)
- [Addestramento del modello](#addestramento-del-modello)
  - [Addestramento Dante](#addestramento-dante)
  - [Addestramento rapido (debug/test)](#addestramento-rapido-debugtest)
  - [Opzioni di addestramento personalizzate](#opzioni-di-addestramento-personalizzate)
- [Generazione di testo e SMILES](#generazione-di-testo-e-smiles)
  - [Generazione testo dantesco](#generazione-testo-dantesco)
  - [Generazione SMILES](#generazione-smiles)
- [Esempi di utilizzo](#esempi-di-utilizzo)
- [Consigli e troubleshooting](#consigli-e-troubleshooting)
- [Riferimenti](#riferimenti)

---

## Caratteristiche
- **Quantum Transformer**: architettura ibrida che integra layer quantistici custom in uno stack transformer.
- **Doppia modalità**: generazione di testo (italiano, stile Dante) e generazione di stringhe SMILES.
- **Addestramento custom**: training su dataset testuale (Inferno di Dante) o dataset molecolare.
- **Checkpoints separati**: salvataggio e caricamento di modelli per task diversi.
- **Configurazione flessibile**: parametri di training e generazione facilmente personalizzabili da CLI.

---

## Requisiti
- Python 3.8+
- PyTorch
- CUDA (opzionale, consigliato per training su GPU)
- [cudaq](https://github.com/NVIDIA/cuda-quantum) (per layer quantistici)
- tqdm, numpy, requests (per logging, dataset, download)
- RDKit (solo per generazione SMILES)

Installa le dipendenze principali con:

```bash
!pip install cudaq
!pip install rdkit==2024.9.4
!pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
!pip install torchvision
!pip install torchaudio
!pip install pandas==2.2.2
!pip install torchdata==0.10.1
!pip install tqdm==4.67.1
!pip install scikit-learn==1.5.1
!pip install seaborn==0.13.2
!pip install gdown==5.2.0
```

---

## Struttura del progetto

```
Quantum-Transformer/
│
├── main.py                  # Entry point principale (training/generazione)
├── run_dante.py             # Script comodo per training Dante
├── reproduce.py             # (placeholder per riproducibilità)
├── dataset/
│   └── inferno.txt          # Testo dell'Inferno di Dante (già incluso)
├── model_checkpoints/
│   ├── dante/               # Checkpoint modelli Dante
│   └── smile/               # Checkpoint modelli SMILES
├── src/
│   ├── dante_trainer.py     # Logica di training per Dante
│   ├── quantum_layer.py     # Layer quantistici custom
│   ├── quantum_transformer.py # Architettura modello
│   ├── utils.py             # Utility (tokenizzazione, generazione, ecc.)
│   └── unitary_library.py   # Kernel quantistici CUDAQ
└── README.md                # Questo file
```

---

## Dataset
- **Testo**: `dataset/inferno.txt` (già incluso, nessun download necessario)
- **SMILES**: (non incluso, puoi integrare il tuo dataset)

---

## Addestramento del modello

### Addestramento Dante
Per addestrare il modello sul testo dell'Inferno di Dante:

```bash
python main.py trainDante
```

Oppure, usando lo script dedicato (identico):

```bash
python run_dante.py
```

#### Opzioni principali:
- `--epochs N`         Numero di epoche (default: 20)
- `--batch_size N`     Batch size (default: 16)
- `--block_size N`     Lunghezza sequenza (default: 128)
- `--lr VAL`           Learning rate (default: 3e-4)
- `--save_every N`     Salva checkpoint ogni N epoche (default: 5)
- `--fast`             Modalità veloce (per test/debug)

Esempio:

```bash
python main.py trainDante --epochs 50 --batch_size 32 --lr 5e-4
```

### Addestramento rapido (debug/test)
Per un training veloce (utile per testare che tutto funzioni):

```bash
python main.py trainDante --fast
```

---

## Generazione di testo e SMILES

### Generazione testo dantesco
Per generare testo in stile Dante dal modello addestrato:

```bash
python main.py generate --model dante --prompt "Nel mezzo del cammin"
```

Opzioni utili:
- `--max_len N`        Lunghezza massima del testo generato (default: 100)
- `--temperature VAL`  Temperature sampling (default: 0.8)
- `--top_k N`          Top-k sampling (default: 5)

Esempio avanzato:

```bash
python main.py generate --model dante --prompt "Nel mezzo del cammin" --max_len 200 --temperature 0.7 --top_k 40
```

### Generazione SMILES
Per generare una stringa SMILES:

```bash
python main.py generate --model smile --prompt "C"
```

Anche qui puoi personalizzare:

```bash
python main.py generate --model smile --prompt "C" --max_len 50 --temperature 0.8 --top_k 5
```

---

## Esempi di utilizzo

### Dal codice Python

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

## Consigli e troubleshooting
- **Errore shape gradienti**: Se durante l'addestramento ricevi errori su shape dei gradienti, assicurati che la funzione backward in `src/quantum_layer.py` restituisca gradienti con shape `[batch_size, block_size, block_size, groups, param_count]`.
- **CUDA**: Se hai una GPU, assicurati che PyTorch la rilevi (`torch.cuda.is_available()`).
- **Checkpoint**: I modelli addestrati vengono salvati in `model_checkpoints/dante/` e `model_checkpoints/smile/`.
- **Dataset**: Per Dante, il file `inferno.txt` deve essere presente in `dataset/`. Per SMILES, integra il tuo dataset e la relativa logica di training.

---

## Riferimenti
- [cuda-quantum (NVIDIA)](https://github.com/NVIDIA/cuda-quantum)
- [PyTorch](https://pytorch.org/)
- [RDKit](https://www.rdkit.org/)
- Dante Alighieri, "La Divina Commedia" (Project Gutenberg)

---

Per domande, bug o suggerimenti, apri una issue su GitHub o contatta i maintainer del progetto.
