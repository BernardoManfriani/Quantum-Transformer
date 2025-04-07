import torch
from torch import Tensor, nn
from typing import Optional, Tuple
import numpy as np
from src.quantum_layer import AttentionQuantumLayer

class QuantumTransformerModel(nn.Module):
    """
    Modello Transformer Quantistico senza gestione dataset e senza parte condizionale.

    Args:
        qpu_count (int): Numero di QPU (usato da AttentionQuantumLayer).
        vocab_size (int): Dimensione del vocabolario.
        embed_dim (int): Dimensione dell'embedding classico (usato per la matrice Value).
        block_size (int): Lunghezza massima della sequenza.
        num_qubits (int): Numero totale di qubit per l'attenzione quantistica.
        ansatz_layers (int): Numero di layer nell'ansatz quantistico.
        quantum_gradient_method (str): Metodo per il gradiente quantistico (es. 'spsa', 'param-shift').
        epsilon (float): Epsilon per SPSA (se usato).
    """
    def __init__(
        self,
        qpu_count: int,
        vocab_size: int,
        embed_dim: int = 64,
        block_size: int = 22,
        num_qubits: int = 6,
        ansatz_layers: int = 1,
        quantum_gradient_method: str = "spsa",
        epsilon: float = 0.01,
    ):
        super().__init__()
        print("Inizializzazione QuantumTransformerModel...")

        # I qubit sono divisi tra embedding dei token e embedding posizionali
        # Dato che non c'è addestramento condizionale, dividiamo per 2.
        if num_qubits % 2 != 0:
            raise ValueError("Il numero di qubit deve essere pari per essere diviso equamente tra token e posizione.")
        num_qubits_per_register = num_qubits // 2
        # print(f"Qubit per registro (token/posizione): {num_qubits_per_register}")

        # --- Inizializzazione Embedding Quantistici (Parametri/Angoli) ---
        print("Inizializzazione parametri embedding quantistici...")
        self._initialize_quantum_embeddings(
            vocab_size,
            ansatz_layers,
            num_qubits_per_register,
            block_size,
        )

        # --- Inizializzazione Embedding Classici (per Matrice Value) ---
        # Questi sono necessari anche nel modello quantistico per calcolare la matrice V
        print("Inizializzazione embedding classici (per Value)...")
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        # print(f"Embedding Classico Token: {self.token_embed.weight.size()}")
        # print(f"Embedding Classico Posizione: {self.position_embed.shape}")


        # --- Componenti Architettura Transformer ---
        print("Inizializzazione componenti Transformer...")
        self.dropout = nn.Dropout(0.1)
        self.block = QuantumTransformerBlock(
            qpu_count=qpu_count,
            embed_dim=embed_dim,
            num_qubits=num_qubits,
            ansatz_layers=ansatz_layers,
            quantum_gradient_method=quantum_gradient_method,
            epsilon=epsilon,
            # conditional_training=False è implicito qui
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        # print(f"LayerNorm: {self.layer_norm.normalized_shape}")
        # print(f"Output Layer: {self.output.weight.shape}")
        print("Inizializzazione QuantumTransformerModel completata.")

    def _initialize_quantum_embeddings(
        self,
        vocab_size,
        ansatz_layers,
        num_qubits_per_register,
        block_size,
    ):
        """Inizializza i parametri (angoli) per gli embedding quantistici."""
        num_params_per_register = ansatz_layers * num_qubits_per_register

        # Parametri quantistici per i token
        self.token_embed_quantum_parameters = nn.Embedding(
            vocab_size, num_params_per_register
        )
        # print(f"Parametri Quantum Token Embedding: {self.token_embed_quantum_parameters.weight.size()}")

        # Scala i parametri iniziali tra 0 e pi
        self._scale_quantum_parameters(self.token_embed_quantum_parameters.weight)

        # Parametri quantistici per la posizione (inizializzati a zero)
        self.position_embed_quantum_parameters = nn.Parameter(
            torch.zeros(1, block_size, num_params_per_register)
        )
        # print(f"Parametri Quantum Position Embedding: {self.position_embed_quantum_parameters.shape}")


    def _scale_quantum_parameters(self, tensor):
        """Scala i parametri quantistici nell'intervallo [0, pi]."""
        with torch.no_grad():
            min_val, max_val = tensor.min(), tensor.max()
            # Evita divisione per zero se tutti i valori sono uguali
            if max_val - min_val > 1e-6:
                scaled_weights = (tensor - min_val) / (max_val - min_val) * np.pi
            else:
                scaled_weights = torch.zeros_like(tensor) # O assegna pi/2 o altro valore
            tensor.copy_(scaled_weights)
        print("Parametri quantistici scalati a [0, pi]")

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Esegue il forward pass del modello.

        Args:
            idx (torch.Tensor): Tensor contenente gli indici dei token di shape (B, T).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits di output e pesi di attenzione.
        """
        # print("Inizio Forward Pass...")
        B, T = idx.size()
        # print(f"Input idx shape: Batch={B}, SequenceLength={T}")

        if T > self.position_embed.size(1) or T > self.position_embed_quantum_parameters.size(1):
             raise ValueError(f"La lunghezza della sequenza di input ({T}) supera la block_size ({self.position_embed.size(1)})")


        # --- Preparazione Input Classico (per Value) ---
        # Combina embedding classici di token e posizione
        x = self.dropout(self.token_embed(idx) + self.position_embed[:, :T, :])
        # print(f"Input classico combinato (x) shape: {x.shape}")

        # --- Preparazione Input Quantistico (Angoli per Query/Key) ---
        # Prendi i parametri quantistici per la posizione e espandili per il batch
        position_embedding_angles = self.position_embed_quantum_parameters[
            :, :T, :
        ].expand(B, -1, -1) # Espande solo sulla dimensione batch
        # print(f"Angoli embedding posizione (espanso per batch) shape: {position_embedding_angles.shape}")


        # Combina i parametri (angoli) per token e posizione
        # Shape attesa per angles: (NumRegistri, Batch, SeqLen, NumParametriPerRegistro)
        # Qui NumRegistri = 2 (token, posizione)
        angles = torch.stack(
            [self.token_embed_quantum_parameters(idx), position_embedding_angles],
            dim=0 # Aggiunge una dimensione all'inizio per separare i registri
        )
        # print(f"Angoli combinati (angles) shape: {angles.shape}")

        # --- Passaggio attraverso il Blocco Transformer ---
        # Passa sia l'input classico 'x' (per V) sia gli angoli 'angles' (per Q/K quantistici)
        x, attn_weight = self.block(x, angles)
        # print(f"Output dal blocco Transformer - x shape: {x.shape}, attn_weight shape: {attn_weight.shape}")


        # --- Layer Norm Finale e Output ---
        x = self.layer_norm(x)
        # print(f"Output dopo LayerNorm finale: {x.shape}")
        logits = self.output(x)
        # print(f"Output finale (logits) shape: {logits.shape}")
        # print("Fine Forward Pass.")

        return logits, attn_weight


class QuantumTransformerBlock(nn.Module):
    """
    Blocco Transformer Decoder con Attenzione Quantistica.

    Args:
        qpu_count (int): Numero di QPU (passato a AttentionQuantumLayer).
        embed_dim (int): Dimensione dell'embedding (usato per LayerNorm, MLP e input/output Value).
        num_qubits (int): Numero di qubit per l'attenzione quantistica.
        ansatz_layers (int): Layer nell'ansatz quantistico.
        quantum_gradient_method (str): Metodo per il gradiente quantistico.
        epsilon (float): Epsilon per SPSA (se usato).
    """
    def __init__(
        self,
        qpu_count: int,
        embed_dim: int = 64,
        num_qubits: int = 6,
        ansatz_layers: int = 1,
        quantum_gradient_method: str = "spsa",
        epsilon: float = 0.01,
    ):
        super().__init__()
        print("Inizializzazione QuantumTransformerBlock...")

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        # print(f"LayerNorm1 shape: {self.layer_norm1.normalized_shape}")
        # print(f"LayerNorm2 shape: {self.layer_norm2.normalized_shape}")


        # Meccanismo di Attenzione (Quantistico)
        self.attention = AttentionQuantumLayer(
            qpu_count=qpu_count,
            embed_dim=embed_dim, # Dimensione input/output per Value
            shift=torch.tensor(torch.pi / 2), # Valore di shift (esempio)
            ansatz_layers=ansatz_layers,
            num_qubits=num_qubits, # Totale qubit usati nel layer di attenzione
            conditional_training=False, # Specificato come non condizionale
            quantum_gradient_method=quantum_gradient_method,
            epsilon=epsilon,
        )
        print("AttentionQuantumLayer inizializzato.")

        # Multi-layer Perceptron (FeedForward)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1),
        )
        # print(f"MLP: input={embed_dim}, hidden={4 * embed_dim}, output={embed_dim}")
        # print("Inizializzazione QuantumTransformerBlock completata.")


    def forward(self, x: Tensor, angles: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass del blocco Transformer.

        Args:
            x (Tensor): Input classico (per Value e connessioni residue), shape (B, T, C).
            angles (Tensor): Input quantistico (angoli per Q/K), shape (NumRegistri, B, T, NumParams).

        Returns:
            Tuple[Tensor, Tensor]: Output del blocco e pesi di attenzione.
        """
        # print("Inizio Forward Pass - QuantumTransformerBlock...")
        # print(f"Input x shape: {x.shape}, Input angles shape: {angles.shape}")


        # Calcolo Attenzione:
        # Passa l'input normalizzato 'x' (per V) e gli 'angles' (per Q/K quantistici)
        attn_output, attn_weight = self.attention(self.layer_norm1(x), angles)
        # print(f"Output da Attention Layer - attn_output shape: {attn_output.shape}, attn_weight shape: {attn_weight.shape}")

        # Prima Connessione Residua (Add)
        x = x + attn_output # Assumendo che attn_output abbia la stessa shape di x
        # print(f"Dopo prima connessione residua - x shape: {x.shape}")


        # MLP (FeedForward)
        mlp_output = self.mlp(self.layer_norm2(x))
        # print(f"Output da MLP - mlp_output shape: {mlp_output.shape}")


        # Seconda Connessione Residua (Add)
        x = x + mlp_output
        # print(f"Dopo seconda connessione residua - x shape: {x.shape}")
        # print("Fine Forward Pass - QuantumTransformerBlock.")

        return x, attn_weight

