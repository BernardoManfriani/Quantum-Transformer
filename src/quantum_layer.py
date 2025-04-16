from typing import Any, Optional, Tuple

import cudaq
import numpy as np
import torch
from cudaq import spin
from torch import Tensor, nn
from torch.autograd import Function

from src.unitary_library import (
    build_sequence_only_circuit,
)
from src.utils import (
    prepare_attention_inputs,
    remove_redundant_circuits,
    repopulate_tensor,
)


class AttentionQuantumLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        qpu_count: int,
        shift: Tensor,
        ansatz_layers: int = 1,
        num_qubits: int = 6,
        quantum_gradient_method: str = "spsa",
        epsilon: float = 0.01,
    ):
        super(AttentionQuantumLayer, self).__init__()

        self.quantum_gradient_method = quantum_gradient_method
        self.shift = shift
        self.epsilon = epsilon

        total_VQC_params = ansatz_layers * num_qubits

        self.quantum_circuit = AttentionQuantumFunction(
            qpu_count=qpu_count,
            num_qubits=num_qubits,
            ansatz_layers=ansatz_layers,
        )

        self.query_angles = nn.Parameter(torch.zeros(1, 1, total_VQC_params))
        self.key_angles = nn.Parameter(torch.zeros(1, 1, total_VQC_params))
        
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self, x: Tensor, angles: Tensor, _: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:

        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        v = self.value(x)

        # Broadcast query and key angles across all tokens in the sequence and batch.
        broadcasted_query_angles = self.query_angles.expand(B, T, -1)
        broadcasted_key_angles = self.key_angles.expand(B, T, -1)

        # Retrieve angles for token and position embeddings
        tok_angles, pos_angles = angles[:2]

        circuit_parameters = prepare_attention_inputs(
            tok_angles,
            pos_angles,
            broadcasted_query_angles,
            broadcasted_key_angles,
        )
        scale_factor = np.sqrt(C)

        attn_weight = (
            AttentionQuantumFunction.apply(
                circuit_parameters,
                self.shift,
                self.quantum_circuit,
                B,
                T,
                self.quantum_gradient_method,
                self.epsilon,
            )
            * scale_factor
        )

        attn_weight = torch.softmax(attn_weight, dim=-1)
        y = self.projection(torch.matmul(attn_weight, v))
    
        return y, attn_weight



class AttentionQuantumFunction(Function):
    def __init__(self, qpu_count: int, num_qubits: int = 6, ansatz_layers: int = 1):

        self.qpu_count = qpu_count
        self.ansatz_layers = ansatz_layers
        self.num_qubits = num_qubits
        self.hamiltonian = spin.z(0)
        self.build_circuit = build_sequence_only_circuit

    def run(self, parameters):
        device = parameters.device

        def param_splits(x):
            return np.array_split(x.cpu().numpy(), self.qpu_count)

        token_1, position_1, token_2, position_2 = map(
            param_splits, (parameters[:, 0, :], parameters[:, 1, :], parameters[:, 4, :], parameters[:, 5, :])
        )
        query_token_register, query_pos_register, key_token_register, key_pos_register = map(
            lambda x: x.cpu().numpy(), (parameters[:, 2, :], parameters[:, 3, :], parameters[:, 6, :], parameters[:, 7, :])
        )

        query_angles = np.array_split(
            np.concatenate((query_token_register, query_pos_register), axis=1),
            self.qpu_count,
        )
        key_angles = np.array_split(
            np.concatenate((key_token_register, key_pos_register), axis=1),
            self.qpu_count,
        )

        ansatz_layers = np.array_split(np.full((parameters.shape[0], 1), self.ansatz_layers, dtype=int), self.qpu_count)
        num_working_qubits = np.array_split(np.full((parameters.shape[0], 1), self.num_qubits, dtype=int), self.qpu_count)
        
        asyncresults = []
        for i in range(self.qpu_count):
            for j in range(token_1[i].shape[0]):
                asyncresults.append(
                    cudaq.observe_async(
                        self.build_circuit,
                        self.hamiltonian,
                        token_1[i][j, :],
                        position_1[i][j, :],
                        query_angles[i][j, :],
                        token_2[i][j, :],
                        position_2[i][j, :],
                        key_angles[i][j, :],
                        ansatz_layers[i][j],
                        num_working_qubits[i][j, :],
                        qpu_id=i,
                    )
                )

        expectations = torch.tensor(
            [r.get().expectation() for r in asyncresults], device=device
        )
        return expectations

    @staticmethod
    def forward(ctx, parameters: Tensor, shift: Tensor, quantum_circuit: "AttentionQuantumFunction", batch_size: int, block_size: int, quantum_gradient_method: str = "spsa", epsilon: float = 0.01) -> Tensor:
        cleaned_circuit_parameters, unique_index_mapping, lower_triangle_indices, upper_triangle_indices = remove_redundant_circuits(parameters)

        ctx.save_for_backward(cleaned_circuit_parameters, unique_index_mapping, lower_triangle_indices, upper_triangle_indices)
        ctx.shift, ctx.quantum_circuit, ctx.batch_size, ctx.block_size = shift, quantum_circuit, batch_size, block_size
        ctx.quantum_gradient_method, ctx.epsilon = quantum_gradient_method, epsilon

        expectations = quantum_circuit.run(cleaned_circuit_parameters)
        return repopulate_tensor(batch_size, block_size, expectations, unique_index_mapping, lower_triangle_indices, upper_triangle_indices).to(cleaned_circuit_parameters.device)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Optional[Tensor], None, None, None, None, None, None, None]:
        cleaned_circuit_parameters, unique_index_mapping, lower_triangle_indices, upper_triangle_indices = ctx.saved_tensors
        shift, quantum_circuit, batch_size, block_size = ctx.shift, ctx.quantum_circuit, ctx.batch_size, ctx.block_size
        quantum_gradient_method, epsilon = ctx.quantum_gradient_method, ctx.epsilon


        _, groups, param_count = cleaned_circuit_parameters.shape
        device = cleaned_circuit_parameters.device
        gradients = torch.zeros_like(cleaned_circuit_parameters)

        if quantum_gradient_method == "spsa":
            delta = (torch.randint(0, 2, cleaned_circuit_parameters.shape, device=device).float() * 2 - 1) * epsilon
            params_plus, params_minus = cleaned_circuit_parameters + delta, cleaned_circuit_parameters - delta
            with torch.no_grad():
                exp_concat = quantum_circuit.run(torch.cat((params_plus, params_minus), dim=0))
                num_circuits = params_plus.size(0)
                exp_plus = exp_concat[:num_circuits]
                exp_minus = exp_concat[num_circuits:]
            exp_diff = (exp_plus - exp_minus).unsqueeze(-1).unsqueeze(-1)
            spsa_gradient_unique = exp_diff.to(device) / (2 * delta)

            gradients_flat = torch.zeros((batch_size * block_size * block_size, groups, param_count), device=device)
            gradients_flat[lower_triangle_indices] = spsa_gradient_unique[unique_index_mapping]
            gradients = gradients_flat.view(batch_size * block_size * block_size, groups, param_count)

        elif quantum_gradient_method == "parameter-shift":
            shift_right_tensors = []
            shift_left_tensors = []

            for i in range(groups):
                for j in range(param_count):
                    shift_right = cleaned_circuit_parameters.clone()
                    shift_right[:, i, j] += shift
                    shift_left = cleaned_circuit_parameters.clone()
                    shift_left[:, i, j] -= shift
                    shift_right_tensors.append(shift_right)
                    shift_left_tensors.append(shift_left)

            all_shift_right = torch.stack(shift_right_tensors).reshape(-1, groups, param_count)
            all_shift_left = torch.stack(shift_left_tensors).reshape(-1, groups, param_count)

            with torch.no_grad():
                all_grad_expectation_right = quantum_circuit.run(all_shift_right).reshape(groups * param_count, -1)
                all_grad_expectation_left = quantum_circuit.run(all_shift_left).reshape(groups * param_count, -1)

            gradient_estimates = (all_grad_expectation_right - all_grad_expectation_left) / (2 * shift)
            gradients_unique = gradient_estimates.T.reshape(-1, groups, param_count)
            gradients_flat = torch.zeros((batch_size * block_size * block_size, groups, param_count), device=device)
            gradients_flat[lower_triangle_indices] = gradients_unique[unique_index_mapping]
            gradients = gradients_flat.view(batch_size * block_size * block_size, groups, param_count)

        return gradients, None, None, None, None, None, None, None
