# Da SMILES a Dante: Conversione di un Transformer Quantistico per la Generazione di Testo

Questo documento spiega le modifiche necessarie per convertire un Transformer Quantistico originariamente progettato per generare molecole SMILES in uno capace di generare testo poetico nello stile della Divina Commedia di Dante.

## 1. Differenze fondamentali tra SMILES e testo

| Aspetto | Molecole SMILES | Testo Dantesco |
|---------|----------------|----------------|
| **Caratteri** | Insieme limitato di simboli chimici e notazioni speciali (~33 token) | Set esteso di caratteri alfabetici, punteggiatura e simboli vari |
| **Vocabolario** | Token specializzati (`[C-]`, `[NH+]`, ecc.) | Caratteri standard (`a`, `b`, `c`, ..., spazi, punteggiatura) |
| **Struttura** | Regole sintattiche rigide di chimica | Grammatica naturale e struttura poetica |
| **Condizionamento** | Proprietà fisico-chimiche (LogP, MW, ecc.) | Non necessario (generazione non condizionata) |

## 2. Modifiche nella Gestione del Dataset

### Classe `Transformer_Dataset`

```python
# Versione SMILES
def __init__(self, data_path, block_size=None):
    self.data = pd.read_csv(data_path)
    self.smiles = self.data['smiles'].values
    # Vocabolario specializzato per chimica
    self.vocab = ["<pad>", "C", "N", "(", ")", "O", "=", ... ]

# Versione Testo
def __init__(self, data_path=None, block_size=None):
    # Carica il testo completo
    with open(data_path, 'r', encoding='utf-8') as f:
        self.text = f.read()
        
    # Crea vocabolario a livello di carattere
    chars = sorted(list(set(self.text)))
    self.vocab = ["<pad>", "[CLS]", "[EOS]"] + chars
```

### Metodo `__getitem__`

```python
# Versione SMILES
def __getitem__(self, idx):
    # Prende una molecola SMILES specifica
    smiles = self.smiles[idx]
    x, y = self.tokenize_smiles(smiles)
    # Ottiene proprietà fisico-chimiche
    physchem_props = get_physchem_properties(smiles)
    return x, y, physchem_props

# Versione Testo
def __getitem__(self, idx):
    # Estrae una sequenza dal testo
    chunk = self.tokenized_text[idx:idx + self.block_size]
    
    # Input: [CLS] + resto testo
    x = [self.stoi["[CLS]"]] + chunk[:-1]
    # Target: sequenza originale
    y = chunk
    
    # Placeholder zero per compatibilità
    physchem_props = torch.zeros(9, dtype=torch.float32)
    
    return x, y, physchem_props
```

## 3. Modifiche al Forward Pass

### Gestione Proprietà Fisico-Chimiche

```python
# Versione SMILES - Usa proprietà fisico-chimiche
if self.conditional_training:
    physchem_embeddings = (
        self.physchem_embed(physchem_props).unsqueeze(1).expand(-1, T, -1)
    )
    x = self.dropout(
        self.token_embed(idx) + self.position_embed[:, :T, :] + physchem_embeddings
    )

# Versione Testo - Ignora le proprietà fisico-chimiche
x = self.dropout(self.token_embed(idx) + self.position_embed[:, :T, :])
```

### Preparazione Input Quantistico

```python
# Versione SMILES - Includiamo proprietà fisico-chimiche
if self.conditional_training:
    physchem_embeddings_angles = scale_to_range(
        self.physchem_embed_quantum_parameters(physchem_props)
        .unsqueeze(1)
        .expand(-1, T, -1),
        0, torch.pi,
    )
    angles = torch.cat([angles, physchem_embeddings_angles.unsqueeze(0)], dim=0)

# Versione Testo - Solo token e posizione
angles = torch.stack(
    [self.token_embed_quantum_parameters(idx), position_embedding_angles]
)
```

## 4. Funzioni di Generazione

### Funzione `generate_smiles` vs `generate_text`

```python
# generate_smiles
def generate_smiles(model, prompt_smiles, max_len=24, temperature=1.0, top_k=None):
    # Inizializzazione con token SMILES
    prompt_tokens = [cls_token] + tokenize_smiles(prompt_smiles, vocab)
    
    # Generazione con validazione chimica
    with torch.no_grad():
        for _ in range(max_len - len(prompt_tokens)):
            logits, _ = model(idx_cond, physchem_props)  # Include proprietà fisico-chimiche
            # ... logica di generazione ...
            
    # Validazione della molecola risultante
    molecule = Chem.MolFromSmiles(''.join([itos[i] for i in generated_indices]))
    if molecule is None:
        return None  # Molecola non valida
    return ''.join([itos[i] for i in generated_indices])

# generate_text
def generate_text(model, prompt_text, max_len=128, temperature=1.0, top_k=None):
    # Inizializzazione con testo
    prompt_tokens = [cls_token] + [c for c in prompt_text if c in stoi]
    
    # Generazione senza validazione speciale
    with torch.no_grad():
        for _ in range(max_len - len(prompt_tokens)):
            # Proprietà fisico-chimiche vuote
            physchem_props = torch.zeros(9, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(idx_cond, physchem_props)
            # ... logica di generazione ...
            
    # Nessuna validazione speciale necessaria
    return ''.join([itos[i] for i in generated_indices 
                    if itos[i] not in ['[CLS]', '[EOS]', '<pad>']])
```

## 5. Modifiche nell'Architettura

### Attenzione Quantistica

```python
# Versione SMILES - 3 registri per token, posizione, proprietà
num_qubits_per_register = num_qubits // 3  # Se condizionale

# Versione Testo - 2 registri solo per token e posizione
num_qubits_per_register = num_qubits // 2  # Non condizionale
```

## 6. Script di Training

### Training Molecole vs Training Testo

```python
# Training Molecole (reproduce.py)
train_transformer(
    training_data="./dataset/qm9.csv",
    attn_type="quantum",
    conditional_training=True,  # Usa proprietà fisico-chimiche
    epochs=20,
)

# Training Testo (train_text.py / colab_train_dante.py)
train_dante_transformer(
    data_path="./dataset/inferno.txt",
    attn_type="quantum",
    conditional_training=False,  # Non usa proprietà fisico-chimiche
    epochs=10,
)
```

## 7. Implementazioni Ausiliarie

### Tokenizzazione

```python
# Tokenizzazione SMILES
def tokenize_smiles(smiles, vocab):
    # Divide la stringa SMILES in token chimicamente significativi
    # es. tokenize("CC(=O)O") → ["C", "C", "(", "=", "O", ")", "O"]
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|H|Mg|\(|\)|\.|=|#|\\\\|\/|:|~|@|>|<|\+|-|\%[0-9]{2})"
    tokens = re.findall(pattern, smiles)
    return [t for t in tokens]

# Tokenizzazione Testo
def tokenize_text(text, stoi):
    # Semplicemente converte ogni carattere in un indice
    return [stoi[c] for c in text if c in stoi]
```

### Valutazione

```python
# Valutazione Molecole
def evaluate_molecules(generated_smiles):
    # Calcola validità chimica
    valid_mols = [Chem.MolFromSmiles(s) for s in generated_smiles]
    validity_rate = sum(1 for mol in valid_mols if mol is not None) / len(generated_smiles)
    
    # Calcola unicità
    unique_rate = len(set(generated_smiles)) / len(generated_smiles)
    
    # Calcola novità
    training_smiles = set(pd.read_csv("./dataset/train_dataset.csv")["smiles"].tolist())
    novel_rate = sum(1 for s in generated_smiles if s not in training_smiles) / len(generated_smiles)
    
    return validity_rate, unique_rate, novel_rate

# Valutazione Testo (più soggettiva)
def evaluate_text(generated_texts):
    # Qui si potrebbero implementare metriche linguistiche
    # come perplexity, BLEU score, ecc.
    pass
```

## 8. Riassunto delle Modifiche Chiave

1. **Struttura Dati**: Da SMILES CSV a file di testo completo
2. **Vocabolario**: Da token SMILES specializzati a caratteri generici
3. **Tokenizzazione**: Da token chimici a caratteri singoli
4. **Condizionamento**: Rimosso il condizionamento sulle proprietà fisico-chimiche
5. **Proprietà dei Target**: Da validità chimica a qualità linguistiche
6. **Architettura Quantistica**: Semplificata (da 3 a 2 registri di qubit per token e posizione)
7. **Script di Training**: Adattati per dati testuali
8. **Funzioni di Generazione**: Rimosse le validazioni chimiche specifiche

## 9. Vantaggi dell'Architettura Comune

- La struttura fondamentale del Transformer Quantistico rimane invariata
- I meccanismi di attenzione quantistica sono riutilizzabili
- Le funzioni di embedding sono facilmente adattabili
- L'approccio modulare permette di passare da un dominio all'altro con modifiche mirate

Questa conversione dimostra la versatilità dell'architettura Transformer e in particolare dei componenti quantistici, che possono essere applicati sia a problemi di generazione molecolare che a compiti di elaborazione del linguaggio naturale creativo.