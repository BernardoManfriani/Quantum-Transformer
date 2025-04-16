import torch
from torch import Tensor, nn
from typing import Optional, Tuple
import numpy as np
from src.quantum_layer import AttentionQuantumLayer

class QuantumTransformerModel(nn.Module):
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

        num_qubits_per_register = num_qubits // 2

        self._initialize_quantum_embeddings(
            vocab_size,
            ansatz_layers,
            num_qubits_per_register,
            block_size,
        )

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))

        self.dropout = nn.Dropout(0.1)
        self.block = QuantumTransformerBlock(
            qpu_count=qpu_count,
            embed_dim=embed_dim,
            num_qubits=num_qubits,
            ansatz_layers=ansatz_layers,
            quantum_gradient_method=quantum_gradient_method,
            epsilon=epsilon,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def _initialize_quantum_embeddings(
        self,
        vocab_size,
        ansatz_layers,
        num_qubits_per_register,
        block_size,
    ):
        num_params_per_register = ansatz_layers * num_qubits_per_register
        self.token_embed_quantum_parameters = nn.Embedding(
            vocab_size, num_params_per_register
        )

        self._scale_quantum_parameters(self.token_embed_quantum_parameters.weight)

        self.position_embed_quantum_parameters = nn.Parameter(
            torch.zeros(1, block_size, num_params_per_register)
        )

    def _scale_quantum_parameters(self, tensor):
        with torch.no_grad():
            min_val, max_val = tensor.min(), tensor.max()
            scaled_weights = (tensor - min_val) / (max_val - min_val) * np.pi
            tensor.copy_(scaled_weights)

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.size()

        if T > self.position_embed.size(1) or T > self.position_embed_quantum_parameters.size(1):
            raise ValueError(f"La lunghezza della sequenza ({T}) supera la block_size ({self.position_embed.size(1)})")

        x_token = self.token_embed(idx)
        x_pos = self.position_embed[:, :T, :]
        x = self.dropout(x_token + x_pos)

        position_embedding_angles = self.position_embed_quantum_parameters[:, :T, :].expand(B, -1, -1)
        token_angles = self.token_embed_quantum_parameters(idx)
        angles = torch.stack([token_angles, position_embedding_angles], dim=0)

        x, attn_weight = self.block(x, angles)
        x = self.layer_norm(x)
        logits = self.output(x)
    
        return logits, attn_weight

class QuantumTransformerBlock(nn.Module):
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

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)


        self.attention = AttentionQuantumLayer(
            qpu_count=qpu_count,
            embed_dim=embed_dim,
            shift=torch.tensor(torch.pi / 2),
            ansatz_layers=ansatz_layers,
            num_qubits=num_qubits,
            quantum_gradient_method=quantum_gradient_method,
            epsilon=epsilon,
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor, angles: Tensor) -> Tuple[Tensor, Tensor]:
        norm_x = self.layer_norm1(x)
        attn_output, attn_weight = self.attention(norm_x, angles)
        x = x + attn_output
        norm_x2 = self.layer_norm2(x)
        mlp_output = self.mlp(norm_x2)
        x = x + mlp_output

        return x, attn_weight
        return x, attn_weight

