from typing import List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from torch import Tensor

import torch
import torch.nn.functional as F

def scale_to_range(
    tensor: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    """
    Scales a tensor to a specified range.

    Parameters:
    - tensor (torch.Tensor): The tensor to scale.
    - min_val (float): The minimum value of the range.
    - max_val (float): The maximum value of the range.

    Returns:
    - torch.Tensor: The scaled tensor.
    """

    # Prevent division by 0
    if tensor.min() == 0 and tensor.max() == 0:
        return tensor

    # Normalize tensor to [0, 1]
    normalized_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # Scale to [min_val, max_val]
    scaled_tensor = normalized_tensor * (max_val - min_val) + min_val
    return scaled_tensor


def prepare_attention_inputs(
    tok: Tensor,
    pos: Tensor,
    query_weights: Tensor,
    key_weights: Tensor,
    physchem: Optional[Tensor] = None,
) -> Tensor:

    batch_size = tok.size(0)
    seq_len = tok.size(1)
    tok_unsqueezed_i = tok.unsqueeze(2)
    tok_i = tok_unsqueezed_i.expand(-1, -1, seq_len, -1)

    tok_unsqueezed_j = tok.unsqueeze(1)
    tok_j = tok_unsqueezed_j.expand(-1, seq_len, -1, -1)

    pos_unsqueezed_i = pos.unsqueeze(2)
    pos_i = pos_unsqueezed_i.expand(-1, -1, seq_len, -1)

    pos_unsqueezed_j = pos.unsqueeze(1)
    pos_j = pos_unsqueezed_j.expand(-1, seq_len, -1, -1)

    num_groups = 2
    query_splits = torch.chunk(query_weights, num_groups, dim=2)
    key_splits = torch.chunk(key_weights, num_groups, dim=2)
    query_token_i_chunk, query_pos_i_chunk = query_splits[:2]
    key_token_j_chunk, key_pos_j_chunk = key_splits[:2]

    # Espandi query_splits per la dimensione 'j' (indice 2)
    query_token_i = query_token_i_chunk.unsqueeze(2).expand(-1, -1, seq_len, -1)
    query_pos_i = query_pos_i_chunk.unsqueeze(2).expand(-1, -1, seq_len, -1)

    # Espandi key_splits per la dimensione 'i' (indice 1)
    key_token_j = key_token_j_chunk.unsqueeze(1).expand(-1, seq_len, -1, -1)
    key_pos_j = key_pos_j_chunk.unsqueeze(1).expand(-1, seq_len, -1, -1)

    # --- Creazione Lista Tensori Base per lo Stack ---
    input_tensors = [
        tok_i,         # (B, T, T, P_tok)
        pos_i,         # (B, T, T, P_pos)
        query_token_i, # (B, T, T, P_qw_tok)
        query_pos_i,   # (B, T, T, P_qw_pos)
        tok_j,         # (B, T, T, P_tok)
        pos_j,         # (B, T, T, P_pos)
        key_token_j,   # (B, T, T, P_kw_tok)
        key_pos_j,     # (B, T, T, P_kw_pos)
    ]

    param_dim = input_tensors[0].shape[-1] 
    stacking_dim = 3 # La dimensione dove verranno impilati i diversi tipi di tensore (gruppi)
    num_tensors_to_stack = len(input_tensors)
    expected_shape = (batch_size, seq_len, seq_len, num_tensors_to_stack, param_dim)

    final_stacked_tensor = torch.stack(input_tensors, dim=stacking_dim)
    actual_shape = final_stacked_tensor.shape
    return final_stacked_tensor


def remove_redundant_circuits(
    full_batch_parameter_tensor: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Optimizes quantum circuit execution by eliminating redundant evaluations.

    Unlike classical GPUs, where calculating a full attention matrix and applying a mask is efficient,
    in this case it is more efficient to only run quantum simulations that calculate the lower triangle and diagonal explicitly.
    Additionally, duplicate circuits (identical parameter sets) are removed to minimize redundant computations.

    Args:
        full_batch_parameter_tensor (torch.Tensor): Quantum circuit params for each attention matrix element.
                                                    shape (batch, n, n, num_groups, param_count).

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            - unique_lower_triangle_params (Tensor): Unique circuit parameters to be executed.
            - unique_index_mapping (Tensor): Mapping of each non-redundant circuit back to its original index.
            - lower_triangle_indices (Tensor): Indices of circuits that belong to the lower triangle.
            - upper_triangle_indices (Tensor): Indices of circuits that belong to the upper triangle.
    """

    batch_size, n, _, groups, param_count = full_batch_parameter_tensor.shape

    # Create a lower triangular mask for the n x n attention matrix
    mask = torch.tril(
        torch.ones(n, n, dtype=torch.bool, device=full_batch_parameter_tensor.device)
    )
    mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Shape: (1, n, n, 1, 1)
    mask = mask.expand(
        batch_size, n, n, groups, param_count
    )  # Shape: (batch_size, n, n, groups, param_count)

    # Apply the mask: Elements in the upper triangle are marked as -inf to exclude them from execution
    full_batch_parameter_tensor = full_batch_parameter_tensor.clone()
    full_batch_parameter_tensor[~mask.expand(batch_size, n, n, groups, param_count)] = (
        float("-inf")
    )

    # Reshape to the final desired shape: (batch*seq_len^2, groups accepted by our custom cudaq functions, param_count)
    # to ensure proper batching of quantum circuit simulations
    flattened_tensor = full_batch_parameter_tensor.view(
        batch_size * n**2, groups, param_count
    )

    # Identify circuits in the upper triangle by checking if all values are -inf
    is_upper_triangle = torch.isinf(flattened_tensor).all(dim=(1, 2))
    lower_triangle_indices = torch.nonzero(~is_upper_triangle, as_tuple=True)[0]
    upper_triangle_indices = torch.nonzero(is_upper_triangle, as_tuple=True)[0]

    # Extract only the necessary circuits (lower triangle)
    lower_triangle_parameters = flattened_tensor[lower_triangle_indices]

    # Up to this point, there should be batch*(seq_len^2 + seq_len)/2 circuits to run now that we have excluded the upper triangle
    # However, some parameter matrices we give to our custom cudaq functions may be the same,
    # so we want to remove these duplicates to avoid running the same circuit multiple times and instead broadcast the expectation values to the correct positions
    # This massively cuts down on computation time
    unique_lower_triangle_params, unique_index_mapping = torch.unique(
        lower_triangle_parameters, return_inverse=True, dim=0
    )

    return (
        unique_lower_triangle_params,
        unique_index_mapping,
        lower_triangle_indices,
        upper_triangle_indices,
    )


def repopulate_tensor(
    batch_size: int,
    seq_len: int,
    processed_results: Tensor,
    unique_index_mapping: Tensor,
    lower_triangle_indices: Tensor,
    upper_triangle_indices: Tensor,
) -> Tensor:
    """
    Repopulates the original attention tensor from processed results.

    This function reconstructs the original (batch, seq_len, seq_len) tensor by mapping
    processed expectation values back to their correct positions, restoring both unique
    and masked (-inf) values.

    Args:
        batch_size (int): Batch size.
        seq_len (int): Sequence length.
        processed_results (Tensor): Computed expectation values from unique circuits.
        unique_index_mapping (Tensor): Maps all matrices to their unique counterpart's index.
        lower_triangle_indices (Tensor): Indices of valid matrices in the flattened tensor.
        upper_triangle_indices (Tensor): Indices where the matrices were set to -inf.

    Returns:
        Tensor: Reconstructed tensor of shape (batch, seq_len, seq_len).
    """

    device = processed_results.device

    # Initialize the reconstructed tensor with -inf placeholders
    reconstructed_tensor = torch.full(
        (batch_size * seq_len * seq_len,), float("-inf"), device=device
    )

    # Map processed results back to the appropriate positions
    reconstructed_tensor[lower_triangle_indices] = processed_results[
        unique_index_mapping
    ]

    # Explicitly set upper triangle indices to -inf (already set, but ensures robustness)
    reconstructed_tensor[upper_triangle_indices] = float("-inf")

    # Reshape to the original (batch, seq_len, seq_len) format
    return reconstructed_tensor.view(batch_size, seq_len, seq_len)


def generate_smiles(model, prompt_smiles, max_len=24, temperature=1.0, top_k=None):
    """
    Genera una sequenza SMILES autoregressivamente partendo da un prompt.

    Args:
        model: Il modello QuantumTransformerModel addestrato.
        prompt_smiles (str): La stringa SMILES iniziale (es. "CC").
        max_len (int): La lunghezza massima della sequenza generata (inclusi [CLS] e prompt).
        temperature (float): Fattore per riscalare i logits prima del softmax. Valori > 1 aumentano la casualità, < 1 la riducono.
        top_k (int, optional): Se specificato, considera solo i top_k token più probabili per il campionamento.

    Returns:
        str: La stringa SMILES generata.
    """
    model.eval()
    checkpoint = torch.load('/content/drive/MyDrive/Università/QuantumML/Quantum-Transformer/model_checkpoints/quantum_sequence/model_epoch_20.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    device = next(model.parameters()).device

    vocab = ['#', '(', ')', '-', '1', '2', '3', '4', '5', '<pad>', '=', 'C', 'F', 'N', 'O', '[C-]', '[CH-]', '[CLS]', '[EOS]', '[N+]', '[N-]', '[NH+]', '[NH2+]', '[NH3+]', '[O-]', '[c-]', '[cH-]', '[n-]', '[nH+]', '[nH]', 'c', 'n', 'o']

    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    vocab_size = len(vocab)

    # Indici speciali
    cls_token = '[CLS]'
    eos_token = '[EOS]'
    pad_token = '<pad>'
    cls_idx = stoi[cls_token]
    eos_idx = stoi[eos_token]
    pad_idx = stoi[pad_token]

    # 1. Prepara il prompt iniziale
    prompt_tokens = [cls_token] + tokenize_smiles(prompt_smiles, vocab)
    idx = torch.tensor([stoi[token] for token in prompt_tokens], dtype=torch.long, device=device).unsqueeze(0) # Shape: (1, T_prompt)

    # print(f"Prompt iniziale (indici): {idx}")
    # print(f"Prompt iniziale (token): {[itos[i.item()] for i in idx[0]]}")


    # 2. Loop di generazione
    with torch.no_grad(): # Non serve calcolare gradienti
        for _ in range(max_len - len(prompt_tokens)): # Genera fino a max_len token totali
            # Se la sequenza corrente supera block_size, prendi solo gli ultimi block_size token
            # Questo è cruciale perché il modello ha un embedding posizionale fisso
            block_size = model.position_embed.size(1) # Ottieni block_size dal modello
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            print(f"Input al modello (step {_}, shape {idx_cond.shape}): {[itos[i.item()] for i in idx_cond[0]]}")

            # 3. Ottieni i logits dal modello
            logits, _ = model(idx_cond) # Shape logits: (1, T_current, vocab_size)

            # 4. Concentrati sull'ultimo token predetto
            logits = logits[:, -1, :] # Shape: (1, vocab_size)

            # 5. Applica temperature scaling (opzionale, ma utile)
            if temperature != 1.0:
                logits = logits / temperature

            # 6. Applica Top-k (opzionale)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # Imposta a -infinito i logits non top-k

            # 7. Calcola le probabilità
            probs = F.softmax(logits, dim=-1) # Shape: (1, vocab_size)

            # 8. Campiona il token successivo
            idx_next = torch.multinomial(probs, num_samples=1) # Shape: (1, 1)

            # 9. Aggiungi il token campionato alla sequenza
            idx = torch.cat((idx, idx_next), dim=1) # Shape: (1, T_current + 1)

            # 10. Controlla se è stato generato [EOS]
            if idx_next.item() == eos_idx:
                print("Generato token [EOS]. Interruzione.")
                break
            else:
                print(f"Generato token {itos[idx_next.item()]}")


    # 11. Decodifica la sequenza finale
    generated_indices = idx[0].tolist()
    generated_tokens = [itos[i] for i in generated_indices]
    return "".join(generated_tokens) # O return generated_tokens se preferisci la lista
