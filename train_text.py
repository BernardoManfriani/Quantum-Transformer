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
    save_every_n_batches: int = 10,
    attn_type: str = "classical",  # Default to "classical" for speed
    epochs: int = 5,               # Reduced epochs
    batch_size: int = 128,         # Increased batch size
    learning_rate: float = 1e-3,   # Increased learning rate for faster convergence
    weight_decay: float = 0.01,
    block_size: int = 64,          # Reduced block size
    data_path: str = "./dataset/inferno_small.txt",  # Use smaller dataset by default
    device: str = "gpu",
    qpu_count: int = 1,            # Default to single QPU
    seed: int = 42,
    quantum_mode: str = "ultra_fast",
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
                cudaq.set_target(target)
                print("Modalità quantum ultra-fast attivata (configurazione semplificata)")
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
    
    # Dimensione embedding ridotta per velocità
    embed_dim = 32
    
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
        
        # Validation phase - Solo su un numero ridotto di batch per velocità
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        max_val_batches = min(len(val_loader), 5)  # Limita il numero di batch per validation
        
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


def generate_sample_text(model, dataset, device, max_tokens=50, temperature=1.0):
    """Genera un esempio di testo usando il modello addestrato."""
    model.eval()
    
    # Inizializza con [CLS] token
    context = torch.tensor([[dataset.stoi["[CLS]"]]], dtype=torch.long).to(device)
    generated_text = ["[CLS]"]
    
    print("\n--- Esempio di testo generato ---")
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Predici il prossimo token
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Campiona dalla distribuzione
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Converti il token in carattere
            next_char = dataset.itos[next_token.item()]
            
            # Termina se [EOS] o <pad>
            if next_char == "[EOS]" or next_char == "<pad>":
                break
                
            # Aggiungi al testo generato
            generated_text.append(next_char)
            
            # Aggiorna il contesto
            context = torch.cat((context, next_token), dim=1)
    
    # Converti in stringa e stampa
    result = "".join(token for token in generated_text if token not in ["[CLS]", "[EOS]", "<pad>"])
    print(result)
    print("----------------------------")
    
    return result


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