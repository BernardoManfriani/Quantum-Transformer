import argparse
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.quantum_transformer import QuantumTransformerModel
from src.utils import tokenize_text, generate_text

# Parametri di default per l'addestramento
BATCH_SIZE = 16
SEQ_LENGTH = 64
VOCAB_SIZE = 78  # Per il testo italiano con caratteri speciali
EMBED_DIM = 64
BLOCK_SIZE = 128
NUM_QUBITS = 6
ANSATZ_LAYERS = 1
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
CHECKPOINT_DIR = './model_checkpoints'
QPU_COUNT = 1

class InfernoDataset(Dataset):
    """Dataset per il testo dell'Inferno di Dante"""
    
    def __init__(self, text_path: str, block_size: int):
        self.block_size = block_size
        
        # Definizione del vocabolario italiano con caratteri speciali
        self.vocab = [
            '<pad>', '[CLS]', '[EOS]',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'à', 'è', 'é', 'ì', 'ò', 'ù',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'À', 'È', 'É', 'Ì', 'Ò', 'Ù',
            ' ', ',', '.', ';', ':', '!', '?', '-', '\"', '\'', '(', ')'
        ]
        
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        
        # Carica il testo completo
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokens = self.tokenize(text)
        print(f"Testo caricato: {len(self.tokens)} token")
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenizza il testo in caratteri e li converte in indici"""
        chars = []
        
        # Aggiungiamo [CLS] all'inizio di ogni testo
        chars.append(self.stoi['[CLS]'])
        
        for char in text:
            if char in self.stoi:
                chars.append(self.stoi[char])
            else:
                # Per caratteri non nel vocabolario, usiamo il token sconosciuto o ignoriamo
                pass
        
        return chars
    
    def __len__(self):
        # Numero di sequenze possibili nel testo (con potenziale sovrapposizione)
        return max(0, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prende una sequenza di lunghezza block_size come input
        x = torch.tensor(self.tokens[idx:idx + self.block_size], dtype=torch.long)
        
        # L'obiettivo è prevedere il token successivo per ogni posizione
        y = torch.tensor(self.tokens[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        
        return x, y

def train_model_dante(args):
    """Addestra il modello Quantum-Transformer sul testo dell'Inferno"""
    
    # Impostazione del device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Creazione del dataset e dataloader
    dataset = InfernoDataset(args.text_path, args.block_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True  # Elimina l'ultimo batch se incompleto
    )
    
    # Inizializzazione del modello
    model = QuantumTransformerModel(
        qpu_count=args.qpu_count,
        vocab_size=len(dataset.vocab),
        embed_dim=args.embed_dim,
        block_size=args.block_size,
        num_qubits=args.num_qubits,
        ansatz_layers=args.ansatz_layers,
        quantum_gradient_method='spsa',
        epsilon=0.01
    )
    model.to(device)
    
    # Ottimizzatore e funzione di perdita
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(dataloader)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.stoi['<pad>'])
    
    # Creazione della directory per i checkpoint se non esiste
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Ciclo di addestramento
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(x)  # Otteniamo sia i logits che gli attention weights
            
            # Calcolo della loss
            # Reshape logits: [batch_size, block_size, vocab_size] -> [batch_size * block_size, vocab_size]
            logits_flat = logits.view(-1, len(dataset.vocab))
            targets_flat = y.view(-1)
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            
            # Clipping del gradiente per stabilità
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Aggiornamento della loss totale
            total_loss += loss.item()
            
            # Stampa progressione ogni N batch
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Stampa riassunto dell'epoca
        avg_loss = total_loss / len(dataloader)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} completata in {elapsed_time:.2f}s | "
              f"Loss media: {avg_loss:.4f}")
        
        # Salvataggio del modello
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        print(f"Checkpoint salvato: {checkpoint_path}")
        
        # Aggiornamento del miglior modello
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(args.checkpoint_dir, "best_dante_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, best_model_path)
            print(f"Nuovo miglior modello salvato con loss: {best_loss:.4f}")
        
        # Generazione di esempi
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            generate_examples(model, dataset, device)

def generate_examples(model, dataset, device, num_examples=3, max_tokens=50):
    """Genera alcuni esempi dal modello corrente per verificare il progresso"""
    model.eval()
    
    print("\n=== Esempi di generazione ===")
    
    # Prompt di esempio
    prompts = [
        "Nel mezzo del cammin",
        "Amor, ch'a nullo amato",
        "Per me si va ne"
    ]
    
    for i, prompt in enumerate(prompts[:num_examples]):
        # Tokenizza il prompt
        prompt_tokens = ["[CLS]"] + list(prompt)
        input_ids = [dataset.stoi[token] if token in dataset.stoi else random.choice(list(dataset.stoi.values())) for token in prompt_tokens]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Tronca l'input se supera block_size
                if input_tensor.size(1) > dataset.block_size:
                    input_tensor = input_tensor[:, -dataset.block_size:]
                
                # Forward pass
                logits, _ = model(input_tensor)
                
                # Prendi l'ultimo token
                next_token_logits = logits[0, -1, :]
                
                # Applica temperatura per controllo casualità
                temperature = 0.7
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                
                # Campiona il prossimo token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Aggiungi alla sequenza
                input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
                
                # Se è [EOS] o stiamo generando abbastanza, termina
                if next_token.item() == dataset.stoi['[EOS]']:
                    break
        
        # Decodifica la sequenza generata
        output_ids = input_tensor[0].tolist()
        output_text = ''.join([dataset.itos[token_id] for token_id in output_ids if token_id in dataset.itos])
        output_text = output_text.replace('[CLS]', '').replace('[EOS]', '')
        
        print(f"\nPrompt {i+1}: \"{prompt}\"")
        print(f"Testo generato: \"{output_text}\"")
    
    print("\n")