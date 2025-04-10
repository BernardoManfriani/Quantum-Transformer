import argparse
import os
import random
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from src.transformer import Transformer_Dataset, Transformer_Model


def train_dante_transformer(
    checkpoint_dir: str = "./dante_checkpoints",
    save_every_n_batches: int = 10,
    attn_type: str = "quantum",  # "quantum" o "classical"
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    block_size: int = 128,
    device: str = "gpu",
    qpu_count: int = -1,
    seed: int = 42,
):
    """
    Addestramento di un modello Transformer (quantistico o classico) sul testo dell'Inferno di Dante.

    Args:
        checkpoint_dir: Directory dove salvare i checkpoint del modello.
        save_every_n_batches: Ogni quanti batch salvare il checkpoint.
        attn_type: Tipo di attenzione: "quantum" per attenzione quantistica, "classical" per attenzione classica.
        epochs: Numero di epoche.
        batch_size: Dimensione del batch.
        learning_rate: Learning rate.
        weight_decay: Weight decay per la regolarizzazione.
        block_size: Dimensione massima della sequenza.
        device: Device per l'addestramento: "cpu" o "gpu".
        qpu_count: Numero di GPU da utilizzare (-1 = tutte disponibili).
        seed: Seed per la riproducibilità.

    Returns:
        None
    """
    # Setup riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Setup CUDA 
    if device not in {"cpu", "gpu"}:
        raise ValueError("Device deve essere 'cpu' o 'gpu'.")
    
    device_torch = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    
    # Configurazione quantum target se serve
    effective_qpu_count = qpu_count
    if attn_type == "quantum":
        import cudaq
        target = (
            "nvidia" 
            if device == "gpu" and cudaq.has_target("nvidia") 
            else "qpp-cpu"
        )
        cudaq.set_target(target, option="mqpu,fp32")
        effective_qpu_count = cudaq.get_target().num_qpus() if qpu_count == -1 else qpu_count
        print(f"Target quantistico: {target}, QPU count: {effective_qpu_count}")
    
    # Preparazione directory dei checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"I checkpoint saranno salvati in: {checkpoint_dir}")
    
    # Caricamento del dataset
    train_dataset = Transformer_Dataset(data_path="./dataset/inferno_small.txt", block_size=block_size)
    vocab_size = len(train_dataset.vocab)
    
    # Divisione training/validation (90%/10%)
    dataset_size = len(train_dataset)
    val_size = min(int(dataset_size * 0.1), 1000)  # Max 1000 esempi di validation
    train_size = dataset_size - val_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Inizializzazione dataloader
    train_loader = StatefulDataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
    )
    
    val_loader = StatefulDataLoader(
        val_data,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
    )
    
    print(f"Dataset totale: {dataset_size} esempi")
    print(f"Training set: {train_size} esempi")
    print(f"Validation set: {val_size} esempi")
    print(f"Dimensione vocabolario: {vocab_size} token")
    
    # Inizializzazione del modello
    classical_attention = (attn_type == "classical")
    model = Transformer_Model(
        qpu_count=effective_qpu_count,
        vocab_size=vocab_size,
        embed_dim=64,  # Dimensione embedding
        block_size=block_size,
        classical_attention=classical_attention,
        num_qubits=6,  # Numero qubit per l'attenzione quantistica
        ansatz_layers=1,  # Layer nell'ansatz quantistico
        conditional_training=False,  # Non usiamo condizioni per il testo
        classical_parameter_reduction=False,
    ).to(device_torch)
    
    print("Modello inizializzato:")
    print(f"Tipo attenzione: {'classica' if classical_attention else 'quantistica'}")
    print(f"Dimensione embedding: 64")
    print(f"Block size: {block_size}")
    print(f"Numero qubit: {6}")
    
    # Inizializzazione ottimizzatore
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    
    # Scheduler di learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Liste per tracciare le loss
    training_losses = []
    val_losses = []
    
    # Loop di addestramento
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        # Progress bar per il training
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        
        for batch_idx, (x, y, _) in pbar:
            # Sposta i dati sul device
            x, y = x.to(device_torch), y.to(device_torch)
            
            # Azzera i gradienti
            optimizer.zero_grad()
            
            # Forward pass (non serve passare proprietà fisico-chimiche)
            logits, _ = model(x)
            
            # Calcolo della loss (Cross Entropy)
            # Reshape logits: (B, T, vocab_size) -> (B*T, vocab_size)
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            y_flat = y.view(-1)
            
            # Maschera per ignorare i padding (-1)
            mask = y_flat != -1
            if mask.sum() == 0:
                continue  # Salta questo batch se non ci sono token validi
                
            filtered_logits = logits_flat[mask]
            filtered_targets = y_flat[mask]
            
            # Calcolo della loss con i token filtrati
            loss = F.cross_entropy(filtered_logits, filtered_targets)
            
            # Backward pass
            loss.backward()
            
            # Aggiornamento parametri
            optimizer.step()
            
            # Aggiornamento metriche
            total_train_loss += loss.item()
            num_batches += 1
            
            # Aggiornamento della progress bar
            pbar.set_postfix(loss=loss.item())
            
            # Salvataggio checkpoint ogni n batch
            if (batch_idx + 1) % save_every_n_batches == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"model_epoch_{epoch+1}_batch_{batch_idx+1}.pt"
                )
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    training_losses,
                    val_losses,
                    {
                        "seed": seed,
                        "attn_type": attn_type,
                        "classical_attention": classical_attention,
                        "num_qubits": 6,
                        "ansatz_layers": 1,
                        "conditional_training": False,
                        "classical_parameter_reduction": False,
                    },
                )
        
        # Calcolo della loss media dell'epoca
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        training_losses.append(avg_train_loss)
        
        print(f"Training Loss: {avg_train_loss:.6f}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device_torch), y.to(device_torch)
                
                # Forward pass
                logits, _ = model(x)
                
                # Calcolo della loss come nel training
                B, T, C = logits.shape
                logits_flat = logits.view(-1, C)
                y_flat = y.view(-1)
                
                mask = y_flat != -1
                if mask.sum() == 0:
                    continue
                    
                filtered_logits = logits_flat[mask]
                filtered_targets = y_flat[mask]
                
                loss = F.cross_entropy(filtered_logits, filtered_targets)
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        # Calcolo della loss media di validation
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Aggiornamento learning rate
        scheduler.step()
        
        # Salvataggio checkpoint finale dell'epoca
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            training_losses,
            val_losses,
            {
                "seed": seed,
                "attn_type": attn_type,
                "classical_attention": classical_attention,
                "num_qubits": 6,
                "ansatz_layers": 1,
                "conditional_training": False,
                "classical_parameter_reduction": False,
            },
        )
    
    print("\nAddestramento completato!")
    best_epoch = val_losses.index(min(val_losses)) + 1
    print(f"Miglior modello: Epoch {best_epoch} (validation loss: {min(val_losses):.6f})")

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    training_losses: List[float],
    val_losses: List[float],
    config_dict: Dict[str, Union[str, bool, int, float]],
):
    """
    Salva un checkpoint del modello.
    
    Args:
        path: Percorso dove salvare il checkpoint.
        model: Modello da salvare.
        optimizer: Ottimizzatore da salvare.
        scheduler: Scheduler da salvare.
        epoch: Numero dell'epoca corrente.
        training_losses: Lista delle loss di training.
        val_losses: Lista delle loss di validation.
        config_dict: Dizionario con la configurazione del modello.
        
    Returns:
        None
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "training_losses": training_losses,
            "val_losses": val_losses,
            "training_configuration": config_dict,
        },
        path,
    )
    print(f"Checkpoint salvato: {path}")

def train_dante_transformer_fast(
    checkpoint_dir: str = "./dante_fast_checkpoints",
    save_every_n_batches: int = 50,  # Aumentato per ridurre il salvataggio
    attn_type: str = "classical",
    epochs: int = 2,                # Ridotto a 2 epoche come richiesto
    batch_size: int = 256,          # Aumentato a 256 per velocità
    learning_rate: float = 5e-3,    # Aumentato il learning rate per convergenza più rapida
    weight_decay: float = 0.001,    # Ridotto weight decay
    block_size: int = 32,           # Ridotto ulteriormente block size
    data_path: str = "./dataset/inferno_small.txt",
    device: str = "gpu",
    qpu_count: int = 1,
    seed: int = 42,
    quantum_mode: str = "ultra_fast",
    epsilon: float = 0.05,          # Aumentato epsilon per convergenza più veloce
):
    """
    Versione ultra-veloce dell'addestramento di un Transformer su dataset ridotto.
    
    Args:
        checkpoint_dir: Directory dove salvare i checkpoint.
        save_every_n_batches: Ogni quanti batch salvare checkpoint.
        attn_type: "quantum" o "classical".
        epochs: Numero di epoche di addestramento.
        batch_size: Dimensione del batch.
        learning_rate: Learning rate iniziale.
        weight_decay: Weight decay per la regolarizzazione.
        block_size: Dimensione massima della sequenza.
        data_path: Percorso del dataset da utilizzare.
        device: "cpu" o "gpu".
        qpu_count: Numero di QPU (per quantum mode).
        seed: Seed per la riproducibilità.
        quantum_mode: Modalità di simulazione quantistica.
        epsilon: Parametro per la convergenza più veloce.
        
    Returns:
        None
    """
    start_time = time.time()
    
    # Setup riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Setup device
    if device not in {"cpu", "gpu"}:
        raise ValueError("Device deve essere 'cpu' o 'gpu'.")
    
    device_torch = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Utilizzo device: {device_torch}")
    
    # Configurazione quantum target se richiesto
    effective_qpu_count = qpu_count
    if attn_type == "quantum":
        try:
            import cudaq
            target = (
                "nvidia" 
                if device == "gpu" and cudaq.has_target("nvidia") 
                else "qpp-cpu"
            )
            # Attiva modalità ultra veloce se richiesta
            if quantum_mode == "ultra_fast":
                # Configurazione semplificata per evitare errori di memoria
                cudaq.set_target(target)  # Senza opzioni aggiuntive per ultra_fast
                print("Modalità quantum ultra-fast attivata (configurazione base)")
            else:
                cudaq.set_target(target, option="mqpu,fp32")
                print("Modalità quantum standard attivata")
                
            effective_qpu_count = 1  # Forza a 1 QPU per evitare errori di memoria
            print(f"Target quantistico: {target}, QPU count forzato a: {effective_qpu_count}")
        except ImportError:
            print("Attenzione: cudaq non disponibile, si procede con backend CPU")
            effective_qpu_count = 1
    
    # Preparazione directory dei checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"I checkpoint saranno salvati in: {checkpoint_dir}")
    
    # Caricamento del dataset
    print(f"Caricamento dataset da {data_path}")
    train_dataset = Transformer_Dataset(data_path=data_path, block_size=block_size)
    vocab_size = len(train_dataset.vocab)
    
    # Divisione training/validation con set di validation ridotto
    dataset_size = len(train_dataset)
    val_size = min(int(dataset_size * 0.05), 200)  # Max 200 esempi per validation
    train_size = dataset_size - val_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Inizializzazione dataloader con pin_memory=False per velocità
    train_loader = StatefulDataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )
    
    val_loader = StatefulDataLoader(
        val_data,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )
    
    print(f"Dataset totale: {dataset_size} esempi")
    print(f"Training set: {train_size} esempi")
    print(f"Validation set: {val_size} esempi")
    print(f"Dimensione vocabolario: {vocab_size} token")
    
    # Dimensione embedding ridotta ulteriormente per velocità
    embed_dim = 16
    
    # Inizializzazione del modello
    classical_attention = (attn_type == "classical")
    model = Transformer_Model(
        qpu_count=effective_qpu_count,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        classical_attention=classical_attention,
        num_qubits=2,  # Ridotto a 2 per velocità
        ansatz_layers=1,
        conditional_training=False,
        classical_parameter_reduction=False,
        epsilon=epsilon,  # Passa epsilon al modello
    ).to(device_torch)
    
    # Calcola e stampa il numero di parametri del modello
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modello inizializzato con {num_params:,} parametri addestrabili")
    print(f"Tipo attenzione: {'classica' if classical_attention else 'quantistica'}")
    print(f"Dimensione embedding: {embed_dim}")
    print(f"Block size: {block_size}")
    print(f"Numero qubit: 2")
    
    # Inizializzazione ottimizzatore con gradient clipping
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),  # Beta2 aumentato per maggiore stabilità
        eps=1e-8,
        weight_decay=weight_decay,
    )
    
    # Scheduler di learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Liste per tracciare le loss
    training_losses = []
    val_losses = []
    
    # Loop di addestramento
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        # Progress bar per il training
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        
        for batch_idx, (x, y, _) in pbar:
            # Sposta i dati sul device
            x, y = x.to(device_torch), y.to(device_torch)
            
            # Azzera i gradienti
            optimizer.zero_grad()
            
            # Forward pass
            print("Forward...", end="\r")  # Indicatore per debug
            logits, _ = model(x)
            
            # Calcolo della loss (Cross Entropy)
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            y_flat = y.view(-1)
            
            # Maschera per ignorare i padding (-1)
            mask = y_flat != -1
            if mask.sum() == 0:
                continue  # Salta questo batch se non ci sono token validi
            
            # Token filtrati
            filtered_logits = logits_flat[mask]
            filtered_targets = y_flat[mask]
            
            # Calcolo della loss con i token filtrati
            loss = F.cross_entropy(filtered_logits, filtered_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Aggiornamento parametri
            optimizer.step()
            
            # Aggiornamento metriche
            total_train_loss += loss.item()
            num_batches += 1
            
            # Aggiornamento della progress bar
            pbar.set_postfix(loss=loss.item())
            
            # Salvataggio checkpoint ogni n batch (se richiesto)
            if save_every_n_batches > 0 and (batch_idx + 1) % save_every_n_batches == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"model_epoch_{epoch+1}_batch_{batch_idx+1}.pt"
                )
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    training_losses,
                    val_losses,
                    {
                        "seed": seed,
                        "attn_type": attn_type,
                        "classical_attention": classical_attention,
                        "data_path": data_path,
                        "num_qubits": 2,
                        "ansatz_layers": 1,
                        "conditional_training": False,
                        "classical_parameter_reduction": False,
                        "embed_dim": embed_dim,
                    },
                )
                print(f"Checkpoint salvato: {checkpoint_path}")
        
        # Calcolo della loss media dell'epoca
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        training_losses.append(avg_train_loss)
        
        print(f"Training Loss: {avg_train_loss:.6f}")
        
        # Validation phase - Estremamente limitata per velocità
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        max_val_batches = min(len(val_loader), 3)  # Limitato a 3 batch per velocità massima
        
        with torch.no_grad():
            for i, (x, y, _) in enumerate(tqdm(val_loader, desc="Validation", total=max_val_batches)):
                if i >= max_val_batches:
                    break
                    
                x, y = x.to(device_torch), y.to(device_torch)
                
                # Forward pass
                logits, _ = model(x)
                
                # Calcolo della loss come nel training
                B, T, C = logits.shape
                logits_flat = logits.view(-1, C)
                y_flat = y.view(-1)
                
                mask = y_flat != -1
                if mask.sum() == 0:
                    continue
                    
                filtered_logits = logits_flat[mask]
                filtered_targets = y_flat[mask]
                
                loss = F.cross_entropy(filtered_logits, filtered_targets)
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        # Calcolo della loss media di validation
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Tempo impiegato per l'epoca
        epoch_time = time.time() - epoch_start_time
        print(f"Validation Loss: {avg_val_loss:.6f} - Epoch Time: {epoch_time:.2f}s")
        
        # Aggiornamento learning rate
        scheduler.step()
        
        # Salvataggio checkpoint finale dell'epoca
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            training_losses,
            val_losses,
            {
                "seed": seed,
                "attn_type": attn_type,
                "classical_attention": classical_attention,
                "data_path": data_path,
                "num_qubits": 2,
                "ansatz_layers": 1,
                "conditional_training": False,
                "classical_parameter_reduction": False,
                "embed_dim": embed_dim,
            },
        )
        print(f"Checkpoint di fine epoca {epoch+1} salvato!")
    
    total_time = time.time() - start_time
    print(f"\nAddestramento completato in {total_time:.2f} secondi!")
    best_epoch = val_losses.index(min(val_losses)) + 1
    print(f"Miglior modello: Epoch {best_epoch} (validation loss: {min(val_losses):.6f})")

    # Salva un checkpoint finale
    final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pt")
    save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        scheduler,
        epochs-1,
        training_losses,
        val_losses,
        {
            "seed": seed,
            "attn_type": attn_type,
            "classical_attention": classical_attention,
            "data_path": data_path,
            "num_qubits": 2,
            "ansatz_layers": 1,
            "conditional_training": False,
            "classical_parameter_reduction": False,
            "embed_dim": embed_dim,
        },
    )
    print(f"Checkpoint finale salvato: {final_checkpoint_path}")

    # Generazione di un piccolo esempio di testo con il modello addestrato
    generate_sample_text(model, train_dataset, device_torch, max_tokens=50)


def generate_sample_text(model, dataset, device, max_tokens=50, temperature=1.0, top_k=None, prompt=None):
    """
    Generate sample text from the trained model.
    
    Args:
        model: The trained Transformer model
        dataset: The dataset containing vocabulary information
        device: The device to run inference on
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (lower = more deterministic)
        top_k: If set, limit sampling to top k most likely tokens
        prompt: Optional starting prompt text, if None uses a default
        
    Returns:
        str: The generated text
    """
    model.eval()
    
    # Get vocabulary mappings
    stoi = dataset.stoi
    itos = dataset.itos
    
    # Start with default or provided prompt
    if prompt is None:
        prompt = "Nel mezzo del"  # Default starting prompt
    
    # Tokenize prompt
    tokens = ["[CLS]"] + [c for c in prompt]
    token_indices = [stoi.get(token, stoi["[CLS]"]) for token in tokens]
    
    # Convert to tensor
    x = torch.tensor(token_indices, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model block size
            block_size = model.position_embed.size(1)
            
            # Trim to block size if needed
            if x.size(1) > block_size:
                x = x[:, -block_size:]
                
            # Forward pass
            logits, _ = model(x)
            
            # Get logits for the next token
            next_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
                
            # Apply top-k sampling if specified
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to get probabilities
            probs = F.softmax(next_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            x = torch.cat((x, next_token), dim=1)
            
            # Check for end of sequence
            if next_token.item() == stoi.get("[EOS]", -1):
                break
    
    # Convert back to text
    tokens = [itos[i.item()] for i in x[0]]
    
    # Remove special tokens
    generated_text = ''.join([t for t in tokens if t not in ["[CLS]", "[EOS]", "<pad>"]])
    
    return generated_text


def emergency_train_dante_transformer(
    checkpoint_dir: str = "./dante_emergency_checkpoints",
    data_path: str = "./dataset/inferno_small.txt",
    epochs: int = 2,
    batch_size: int = 512,  # Batch size molto grande per velocità massima
    learning_rate: float = 1e-2,  # Learning rate aumentato per convergenza rapida
    block_size: int = 32,    # Block size ridotto 
    embed_dim: int = 16,     # Embedding minimo
    device: str = "gpu",
    seed: int = 42,
):
    """
    Versione di emergenza ultra-veloce che forza l'uso di attenzione classica
    e bypassa completamente CUDA Quantum per evitare problemi di memoria.
    
    Non usa mai quantum mode - questa è una soluzione di fallback quando
    il quantum mode continua a causare errori di memoria.
    """
    start_time = time.time()
    
    # Setup riproducibilità
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device_torch = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Utilizzo device: {device_torch}")
    
    # Preparazione directory dei checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"I checkpoint saranno salvati in: {checkpoint_dir}")
    
    # Caricamento del dataset
    print(f"Caricamento dataset da {data_path}")
    train_dataset = Transformer_Dataset(data_path=data_path, block_size=block_size)
    vocab_size = len(train_dataset.vocab)
    
    # Divisione training/validation con set di validation ridotto
    dataset_size = len(train_dataset)
    val_size = min(int(dataset_size * 0.05), 100)  # Ulteriormente ridotto a max 100 esempi
    train_size = dataset_size - val_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Inizializzazione dataloader con pin_memory=False per velocità
    train_loader = StatefulDataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )
    
    val_loader = StatefulDataLoader(
        val_data,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )
    
    print(f"Dataset totale: {dataset_size} esempi")
    print(f"Training set: {train_size} esempi")
    print(f"Validation set: {val_size} esempi")
    print(f"Dimensione vocabolario: {vocab_size} token")
    
    # Inizializzazione del modello - FORZA ATTENZIONE CLASSICA
    classical_attention = True  # FORZATO TRUE
    model = Transformer_Model(
        qpu_count=1,  # Ignorato dato che classical_attention = True
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        classical_attention=classical_attention,
        num_qubits=2,  # Ignorato dato che classical_attention = True
        ansatz_layers=1,
        conditional_training=False,
        classical_parameter_reduction=False,
    ).to(device_torch)
    
    # Calcola e stampa il numero di parametri del modello
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modello inizializzato con {num_params:,} parametri addestrabili")
    print(f"Tipo attenzione: classica (FORZATA)")
    print(f"Dimensione embedding: {embed_dim}")
    print(f"Block size: {block_size}")
    
    # Inizializzazione ottimizzatore con gradient clipping
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0001,  # Ridotto weight decay 
    )
    
    # Scheduler di learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Liste per tracciare le loss
    training_losses = []
    val_losses = []
    
    # Loop di addestramento
    print("\nINIZIO ADDESTRAMENTO DI EMERGENZA CON ATTENZIONE CLASSICA...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        # Progress bar per il training
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        
        for batch_idx, (x, y, _) in pbar:
            # Sposta i dati sul device
            x, y = x.to(device_torch), y.to(device_torch)
            
            # Azzera i gradienti
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(x)
            
            # Calcolo della loss (Cross Entropy)
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            y_flat = y.view(-1)
            
            # Maschera per ignorare i padding (-1)
            mask = y_flat != -1
            if mask.sum() == 0:
                continue  # Salta questo batch se non ci sono token validi
            
            # Token filtrati
            filtered_logits = logits_flat[mask]
            filtered_targets = y_flat[mask]
            
            # Calcolo della loss con i token filtrati
            loss = F.cross_entropy(filtered_logits, filtered_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Aggiornamento parametri
            optimizer.step()
            
            # Aggiornamento metriche
            total_train_loss += loss.item()
            num_batches += 1
            
            # Aggiornamento della progress bar
            pbar.set_postfix(loss=loss.item())
        
        # Calcolo della loss media dell'epoca
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        training_losses.append(avg_train_loss)
        
        print(f"Training Loss: {avg_train_loss:.6f}")
        
        # Validation phase - Solo su un numero ridotto di batch per velocità
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        max_val_batches = min(len(val_loader), 2)  # Limitato a 2 batch per velocità massima
        
        with torch.no_grad():
            for i, (x, y, _) in enumerate(tqdm(val_loader, desc="Validation", total=max_val_batches)):
                if i >= max_val_batches:
                    break
                    
                x, y = x.to(device_torch), y.to(device_torch)
                
                # Forward pass
                logits, _ = model(x)
                
                # Calcolo della loss come nel training
                B, T, C = logits.shape
                logits_flat = logits.view(-1, C)
                y_flat = y.view(-1)
                
                mask = y_flat != -1
                if mask.sum() == 0:
                    continue
                    
                filtered_logits = logits_flat[mask]
                filtered_targets = y_flat[mask]
                
                loss = F.cross_entropy(filtered_logits, filtered_targets)
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        # Calcolo della loss media di validation
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Tempo impiegato per l'epoca
        epoch_time = time.time() - epoch_start_time
        print(f"Validation Loss: {avg_val_loss:.6f} - Epoch Time: {epoch_time:.2f}s")
        
        # Aggiornamento learning rate
        scheduler.step()
    
    # Salva un checkpoint finale
    final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pt")
    save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        scheduler,
        epochs-1,
        training_losses,
        val_losses,
        {
            "seed": seed,
            "attn_type": "classical",  # Sempre classical in modalità emergenza
            "classical_attention": True,
            "data_path": data_path,
            "embed_dim": embed_dim,
        },
    )
    print(f"Checkpoint finale salvato: {final_checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"\nAddestramento completato in {total_time:.2f} secondi!")
    
    # Generazione di un piccolo esempio di testo con il modello addestrato
    generate_sample_text(model, train_dataset, device_torch, max_tokens=50)


def train_dante_fast(
    data_path: str = "./dataset/inferno_small.txt",
    checkpoint_dir: str = "./dante_fast_checkpoints",
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 5e-3,
    embed_dim: int = 32,
    block_size: int = 64,
    device: str = "gpu",
    seed: int = 42
):
    """
    Fast training function for Dante's Inferno text generation.
    Optimized for speed rather than quality.
    
    Args:
        data_path: Path to the dataset file
        checkpoint_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        embed_dim: Embedding dimension
        block_size: Maximum sequence length
        device: Device to run on ('cpu' or 'gpu')
        seed: Random seed for reproducibility
    """
    start_time = time.time()
    print(f"Starting fast training on {data_path}...")
    
    # Set reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device_torch = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device_torch}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Load dataset
    print(f"Loading dataset from {data_path}")
    train_dataset = Transformer_Dataset(data_path=data_path, block_size=block_size)
    vocab_size = len(train_dataset.vocab)
    
    # Split into train/validation
    dataset_size = len(train_dataset)
    val_size = min(int(dataset_size * 0.05), 100)  # 5% for validation, max 100 examples
    train_size = dataset_size - val_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = StatefulDataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )
    
    val_loader = StatefulDataLoader(
        val_data,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )
    
    print(f"Total dataset: {dataset_size} examples")
    print(f"Training set: {train_size} examples")
    print(f"Validation set: {val_size} examples")
    print(f"Vocabulary size: {vocab_size} tokens")
    
    # Initialize model - using classical attention for speed
    model = Transformer_Model(
        qpu_count=1,  # Not used with classical attention
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        classical_attention=True,  # Always use classical attention for speed
        num_qubits=2,  # Not used with classical attention
        ansatz_layers=1,
        conditional_training=False,  # No conditioning for text
        classical_parameter_reduction=False,
        epsilon=0.01,
    ).to(device_torch)
    
    # Count and report parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} trainable parameters")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Block size: {block_size}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),  # Slightly higher beta2 for text
        eps=1e-8,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Track losses
    training_losses = []
    val_losses = []
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        for batch_idx, (x, y, _) in pbar:
            # Move data to device
            x, y = x.to(device_torch), y.to(device_torch)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(x)
            
            # Calculate loss
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            y_flat = y.view(-1)
            
            # Mask out padding tokens
            mask = y_flat != -1
            if mask.sum() == 0:
                continue  # Skip batch if no valid tokens
            
            filtered_logits = logits_flat[mask]
            filtered_targets = y_flat[mask]
            
            # Calculate loss
            loss = F.cross_entropy(filtered_logits, filtered_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update metrics
            total_train_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        training_losses.append(avg_train_loss)
        
        print(f"Training Loss: {avg_train_loss:.6f}")
        
        # Validation phase - limited to save time
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        max_val_batches = min(len(val_loader), 5)  # Limit validation batches
        
        with torch.no_grad():
            for i, (x, y, _) in enumerate(tqdm(val_loader, desc="Validation", total=max_val_batches)):
                if i >= max_val_batches:
                    break
                
                x, y = x.to(device_torch), y.to(device_torch)
                
                # Forward pass
                logits, _ = model(x)
                
                # Calculate loss
                B, T, C = logits.shape
                logits_flat = logits.view(-1, C)
                y_flat = y.view(-1)
                
                mask = y_flat != -1
                if mask.sum() == 0:
                    continue
                
                filtered_logits = logits_flat[mask]
                filtered_targets = y_flat[mask]
                
                loss = F.cross_entropy(filtered_logits, filtered_targets)
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Print epoch stats
        epoch_time = time.time() - epoch_start_time
        print(f"Validation Loss: {avg_val_loss:.6f} - Epoch Time: {epoch_time:.2f}s")
        
        # Update learning rate
        scheduler.step()
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pt")
    save_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        scheduler,
        epochs-1,
        training_losses,
        val_losses,
        {
            "seed": seed,
            "attn_type": "classical",
            "classical_attention": True,
            "data_path": data_path,
            "embed_dim": embed_dim,
            "block_size": block_size,
        },
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    
    # Calculate total training time
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds!")
    
    # Generate a sample text to show results
    print("\n--- Generated Text Sample ---")
    sample_text = generate_sample_text(model, train_dataset, device_torch, max_tokens=50, temperature=0.8)
    print(sample_text)
    return model, train_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestramento Transformer su Inferno di Dante")
    
    # Argomento per scegliere la modalità di addestramento
    parser.add_argument("--fast-mode", action="store_true",
                      help="Usa la modalità di addestramento veloce")
    
    # Parametri comuni
    parser.add_argument("--checkpoint-dir", type=str, default="./dante_checkpoints",
                        help="Directory dove salvare i checkpoint")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Salva il modello ogni N batch")
    parser.add_argument("--attn-type", type=str, choices=["quantum", "classical"],
                        default="classical", help="Tipo di meccanismo di attenzione")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Numero di epoche di addestramento")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Dimensione del batch")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate iniziale")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay per la regolarizzazione")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Dimensione massima della sequenza")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"],
                        default="gpu", help="Device da utilizzare")
    parser.add_argument("--qpu-count", type=int, default=-1,
                        help="Numero di GPU per la simulazione quantistica (-1 = tutte)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed per la riproducibilità")
    
    # Parametri specifici per modalità veloce
    parser.add_argument("--data-path", type=str, default="./dataset/inferno_small.txt",
                        help="Percorso del dataset (usato solo in modalità fast)")
    parser.add_argument("--quantum-mode", type=str, choices=["normal", "ultra_fast"],
                        default="ultra_fast", help="Modalità quantum (solo per attn-type=quantum)")
    
    args = parser.parse_args()
    
    if args.fast_mode:
        print("Avvio addestramento in modalità veloce...")
        # Se in modalità veloce, usa i valori predefiniti ottimizzati per quella funzione
        train_dante_transformer_fast(
            data_path=args.data_path,
            checkpoint_dir=args.checkpoint_dir,
            save_every_n_batches=args.save_every,
            attn_type=args.attn_type,
            epochs=args.epochs if args.epochs < 10 else 5,  # Limita a max 5 epoche in modalità veloce
            batch_size=max(args.batch_size, 128),  # Usa almeno batch size 128 in modalità veloce
            learning_rate=max(args.learning_rate, 1e-3),  # Aumenta il learning rate in modalità veloce
            weight_decay=args.weight_decay,
            block_size=min(args.block_size, 64),  # Riduce block size in modalità veloce
            device=args.device,
            qpu_count=args.qpu_count if args.attn_type == "classical" else 1,
            quantum_mode=args.quantum_mode,
            seed=args.seed,
        )
    else:
        print("Avvio addestramento in modalità standard...")
        train_dante_transformer(
            checkpoint_dir=args.checkpoint_dir,
            save_every_n_batches=args.save_every,
            attn_type=args.attn_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            block_size=args.block_size,
            device=args.device,
            qpu_count=args.qpu_count,
            seed=args.seed,
        )

def train_dante_fast(
    data_path: str = "./dataset/inferno_small.txt",
    checkpoint_dir: str = "./dante_fast_checkpoints",
    epochs: int = 2,
    batch_size: int = 64,
    block_size: int = 64,
    embed_dim: int = 32,
    learning_rate: float = 5e-3,
    seed: int = 42,
    device: str = "gpu",
    save_checkpoint: bool = True,
    generate_sample: bool = True
):
    """
    Train a model on Dante's Inferno text quickly (optimized for speed over quality).
    
    Args:
        data_path: Path to the dataset file
        checkpoint_dir: Directory to save checkpoints
        epochs: Number of training epochs (default: 2 for speed)
        batch_size: Batch size for training
        block_size: Maximum sequence length
        embed_dim: Embedding dimension
        learning_rate: Learning rate
        seed: Random seed for reproducibility
        device: Device to train on ("cpu" or "gpu")
        save_checkpoint: Whether to save checkpoints
        generate_sample: Whether to generate a sample after training
    """
    import time
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device_torch = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    print(f"Using device: {device_torch}")
    
    # Prepare checkpoint directory
    if save_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Load dataset
    print(f"Loading dataset from {data_path}")
    train_dataset = Transformer_Dataset(data_path=data_path, block_size=block_size)
    vocab_size = len(train_dataset.vocab)
    
    # Split train/validation (95%/5%)
    dataset_size = len(train_dataset)
    val_size = min(int(dataset_size * 0.05), 100)  # Small validation set
    train_size = dataset_size - val_size
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = StatefulDataLoader(
        train_data,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )
    
    val_loader = StatefulDataLoader(
        val_data,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=0,
    )
    
    print(f"Total dataset: {dataset_size} examples")
    print(f"Training set: {train_size} examples")
    print(f"Validation set: {val_size} examples")
    print(f"Vocabulary size: {vocab_size} tokens")
    
    # Initialize the model
    model = Transformer_Model(
        qpu_count=1,  # Not using quantum features
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        classical_attention=True,  # Use classical attention for speed
        num_qubits=2,  # Minimal
        ansatz_layers=1,
        conditional_training=False,
        classical_parameter_reduction=False,
        epsilon=0.01,  
    ).to(device_torch)
    
    # Count and print model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} trainable parameters")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Block size: {block_size}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    
    # Initialize scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    training_losses = []
    val_losses = []
    
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        # Progress bar for training
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (x, y, _) in pbar:
            x, y = x.to(device_torch), y.to(device_torch)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(x)
            
            # Calculate loss
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            y_flat = y.view(-1)
            
            # Mask padding tokens
            mask = y_flat != -1
            if mask.sum() == 0:
                continue
            
            # Filter tokens
            filtered_logits = logits_flat[mask]
            filtered_targets = y_flat[mask]
            
            # Compute loss
            loss = F.cross_entropy(filtered_logits, filtered_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Update metrics
            total_train_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item())
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        training_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc="Validation"):
                x, y = x.to(device_torch), y.to(device_torch)
                
                # Forward pass
                logits, _ = model(x)
                
                # Calculate loss
                B, T, C = logits.shape
                logits_flat = logits.view(-1, C)
                y_flat = y.view(-1)
                
                # Mask padding tokens
                mask = y_flat != -1
                if mask.sum() == 0:
                    continue
                
                # Filter tokens
                filtered_logits = logits_flat[mask]
                filtered_targets = y_flat[mask]
                
                # Compute loss
                loss = F.cross_entropy(filtered_logits, filtered_targets)
                
                # Update metrics
                total_val_loss += loss.item()
                num_val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"Time: {epoch_time:.2f}s")
              
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if save_checkpoint:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'training_losses': training_losses,
                'val_losses': val_losses,
                'config': {
                    'seed': seed,
                    'embed_dim': embed_dim,
                    'block_size': block_size,
                    'data_path': data_path,
                }
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Generate sample text
    if generate_sample:
        print("\nGenerating sample text...")
        sample_text = generate_sample_text(
            model, 
            train_dataset, 
            device_torch,
            max_tokens=100,
            temperature=0.8,
            top_k=5
        )
        print(f"\nSample generated text:\n{sample_text}")
    
    return model, train_dataset