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


def train_dante_transformer_lite(
    data_path: str = "./dataset/inferno_small.txt",
    checkpoint_dir: str = "./colab_checkpoints",
    save_every_n_batches: int = 10,  # Cambiato da -1 a 10 per salvare più frequentemente
    attn_type: str = "classical",  # Cambiato default a "classical" per maggiore velocità
    epochs: int = 3,  # Ridotto a 3 epoche
    batch_size: int = 128,  # Aumentato a 128 per velocizzare
    learning_rate: float = 1e-3,  # Aumentato a 1e-3 per convergenza più rapida
    weight_decay: float = 0.01,
    block_size: int = 32,  # Ridotto a 32 (era 64)
    embed_dim: int = 16,  # Ridotto a 16 (era 32)
    device: str = "gpu",
    qpu_count: int = 1,
    seed: int = 42,
    quantum_mode: str = "ultra_fast",  # Nuova opzione per modalità veloce
):
    """
    Versione ultra-ottimizzata per Colab dell'addestramento di un Transformer 
    su un dataset ridotto dell'Inferno di Dante.
    
    Questa versione usa:
    - Dataset molto ridotto
    - Embedding dimension minima
    - Block size ridotta 
    - Learning rate aumentato
    - Salvataggio checkpoint frequente
    - Opzione di modalità quantum ultra veloce
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
                # Usa meno opzioni per evitare problemi di memoria
                cudaq.set_target(target)
                print("Modalità quantum ultra-fast attivata con configurazione semplificata")
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
    
    # Divisione training/validation (90%/10%) - Ridotta la dimensione di validation
    dataset_size = len(train_dataset)
    val_size = min(int(dataset_size * 0.05), 200)  # Ridotto a 5% o max 200 esempi
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
    
    # Inizializzazione del modello
    classical_attention = (attn_type == "classical")
    model = Transformer_Model(
        qpu_count=effective_qpu_count,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        classical_attention=classical_attention,
        num_qubits=2,  # Ridotto a 2 (era 4)
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
        betas=(0.9, 0.95),
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
            print("Forward...", end="\r")  # Aggiunto indicatore per debug
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
        
        # Validation phase - Solo alla fine di ogni epoca per risparmiare tempo
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
    """Salva un checkpoint del modello."""
    try:
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
    except Exception as e:
        print(f"Errore nel salvare il checkpoint: {e}")

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
    parser = argparse.ArgumentParser(description="Training ultraleggero su Inferno di Dante per Colab")
    
    parser.add_argument("--data-path", type=str, default="./dataset/inferno_small.txt",
                        help="Percorso del dataset ridotto")
    parser.add_argument("--checkpoint-dir", type=str, default="./colab_checkpoints",
                        help="Directory dove salvare i checkpoint")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Salva il modello ogni N batch (default=10)")
    parser.add_argument("--attn-type", type=str, choices=["quantum", "classical"],
                        default="classical", help="Tipo di meccanismo di attenzione")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Numero di epoche di addestramento")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Dimensione del batch")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate iniziale")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay per la regolarizzazione")
    parser.add_argument("--block-size", type=int, default=32,
                        help="Dimensione massima della sequenza")
    parser.add_argument("--embed-dim", type=int, default=16,
                        help="Dimensione dell'embedding")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"],
                        default="gpu", help="Device da utilizzare")
    parser.add_argument("--qpu-count", type=int, default=1,
                        help="Numero di GPU per la simulazione quantistica")
    parser.add_argument("--quantum-mode", type=str, choices=["normal", "ultra_fast"],
                        default="ultra_fast", help="Modalità quantum (solo per attn-type=quantum)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed per la riproducibilità")
    
    args = parser.parse_args()
    
    # Esegui l'addestramento con i parametri specificati
    train_dante_transformer_lite(
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_batches=args.save_every,
        attn_type=args.attn_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        device=args.device,
        qpu_count=args.qpu_count,
        quantum_mode=args.quantum_mode,
        seed=args.seed,
    )