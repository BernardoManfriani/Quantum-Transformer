import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import logging
import time

from src.quantum_transformer import QuantumTransformerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Italian vocabulary including special tokens and punctuation
VOCAB = [
    '<pad>', '[CLS]', '[EOS]',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'à', 'è', 'é', 'ì', 'ò', 'ù',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'À', 'È', 'É', 'Ì', 'Ò', 'Ù',
    ' ', ',', '.', ';', ':', '!', '?', '-', '\"', '\'', '(', ')'
]

class DanteDataset(Dataset):
    def __init__(self, text_path, block_size=128):
        """Initialize the Dante dataset from a text file."""
        self.block_size = block_size
        self.vocab = VOCAB
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        
        # Load text
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text file not found at {text_path}")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple tokenization by characters and words
        tokens = self._tokenize_text(text)
        
        # Convert tokens to indices and prepare training examples
        self.data = []
        for i in range(0, len(tokens) - block_size):
            x = tokens[i:i+block_size]
            y = tokens[i+1:i+block_size+1]
            self.data.append((x, y))
        
        logger.info(f"Dataset created with {len(self.data)} samples")
    
    def _tokenize_text(self, text):
        """Tokenize text into characters."""
        # First insert special tokens
        text = '[CLS] ' + text
        
        # Convert to token indices
        result = []
        for char in text:
            if char in self.stoi:
                result.append(self.stoi[char])
            else:
                # Skip characters not in vocabulary
                continue
        
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

def train_dante_model(epochs=20, batch_size=16, block_size=128, learning_rate=3e-4, save_every=5, fast_mode=False):
    """Train the Quantum Transformer model on Dante's Inferno."""
    # Get absolute paths for better reliability
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(base_dir, "model_checkpoints", "dante")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataset paths
    text_path = os.path.join(base_dir, "dataset", "inferno.txt")
    
    # Verify that inferno.txt exists
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Il file {text_path} non esiste. Assicurati di aver copiato inferno.txt nella cartella dataset.")
    else:
        logger.info(f"Utilizzo del file di testo: {text_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    if fast_mode:
        # Use smaller settings for faster training
        epochs = 2
        batch_size = 8
        block_size = 64
        learning_rate = 5e-4
    
    try:
        dataset = DanteDataset(text_path, block_size=block_size)
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True
    )
    
    logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Initialize model
    vocab_size = len(VOCAB)
    embed_dim = 64
    num_qubits = 6
    ansatz_layers = 1
    qpu_count = 1
    
    model = QuantumTransformerModel(
        qpu_count=qpu_count,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        num_qubits=num_qubits,
        ansatz_layers=ansatz_layers,
        quantum_gradient_method='spsa',
        epsilon=0.01
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Training metrics
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training for {epochs} epochs")
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, (x, y) in train_pbar:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for batch_idx, (x, y) in val_pbar:
                x, y = x.to(device), y.to(device)
                
                logits, _ = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                
                total_val_loss += loss.item()
                val_pbar.set_postfix({"loss": loss.item()})
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Report metrics
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'vocab': VOCAB
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_dante_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'vocab': VOCAB,
                'best_val_loss': best_val_loss
            }, best_model_path)
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    logger.info("Training completed!")
    
    # Final save
    final_model_path = os.path.join(checkpoint_dir, "final_dante_model.pt")
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'vocab': VOCAB
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    return model