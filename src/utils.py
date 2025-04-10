from typing import List, Optional, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from torch import Tensor

import torch
import torch.nn.functional as F


def get_physchem_properties(smiles: str) -> List[float]:
    """
    Calculates physicochemical properties for a given molecule.

    Parameters:
    - smiles (str): The SMILES representation of the molecule.

    Returns:
    - List[float]: A list of calculated physicochemical properties for the molecule.
    """

    mol = Chem.MolFromSmiles(smiles)

    properties = [
        ("MW", Descriptors.MolWt(mol)),  # type: ignore
        ("HBA", rdMolDescriptors.CalcNumHBA(mol)),
        ("HBD", rdMolDescriptors.CalcNumHBD(mol)),
        ("nRot", Descriptors.NumRotatableBonds(mol)),  # type: ignore
        ("nRing", rdMolDescriptors.CalcNumRings(mol)),
        ("nHet", rdMolDescriptors.CalcNumHeteroatoms(mol)),
        ("TPSA", Descriptors.TPSA(mol)),  # type: ignore
        ("LogP", Crippen.MolLogP(mol)),  # type: ignore
        ("StereoCenters", len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
    ]

    return [value for _, value in properties]


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
    """
    Prepares attention inputs for batch processing with quantum circuits.

    This function organizes token, position, and key/query tensors of angles into pairwise combinations
    suitable for CUDA-Q functions. It structures input data into (batch, seq_len, seq_len, groups, param_count)
    format for efficient batched execution.


    Args:
        tok (torch.Tensor): Token embeddings (batch, seq_len, ansatz_layers * num_tok_qubits).
        pos (torch.Tensor): Position embeddings (batch, seq_len, ansatz_layers * num_pos_qubits).
        query_weights (torch.Tensor): Query weight tensor (batch, seq_len, ansatz_layers * total_working_qubits).
        key_weights (torch.Tensor): Key weight tensor (batch, seq_len, ansatz_layers * total_working_qubits).
        physchem (torch.Tensor, optional): Physicochemical embeddings (batch, seq_len, ansatz_layers * num_physchem_qubits), if used.

    Returns:
        torch.Tensor: A tensor of shape (batch, seq_len, seq_len, groups, param_count)

                      Notes:
                        - The final two dimensions are tensors [tok1, pos1, query_weights1, tok2, pos2, key_weights2]
                            or [tok1, pos1, query_weights1, tok2, pos2, key_weights2, physchem]
                            for all pairwise combinations

                        - Here, a "group" is a set of parameters that act on the same register (tok, pos, or physchem).

                        - Parameters are grouped by register (which are all of equal size in this work since tok_size == pos_size)
                          rather than by unitary, as grouping by unitaries
                          (which do not have the same number of parameters, i.e. token and position registers are 3 qubits while query and key states are all 6 working qubits)
                          would make it difficult to batch the tensors. It is more difficult work with ragged-arrayed tensors for other operations we used.

                        - These tensors we will eventually need to be flattened from shape (batch, n, n, groups, param_count) to
                          (batch*n*n, groups, param_count) to feed into the custom CUDA-Q functions.


    """

    # Get the sequence length dynamically
    seq_len = tok.size(1)

    # Expand token and position embeddings for pairwise interactions
    tok_i, tok_j = tok.unsqueeze(2).expand(-1, -1, seq_len, -1), tok.unsqueeze(
        1
    ).expand(-1, seq_len, -1, -1)
    pos_i, pos_j = pos.unsqueeze(2).expand(-1, -1, seq_len, -1), pos.unsqueeze(
        1
    ).expand(-1, seq_len, -1, -1)

    # As discussed in the docstring, we break the parameters for the query/key PQCs
    # into groups that act on the token register and ones that act on position register
    # (and possibly the physchem register)
    num_groups = 3 if physchem is not None else 2
    query_splits = torch.chunk(query_weights, num_groups, dim=2)
    key_splits = torch.chunk(key_weights, num_groups, dim=2)

    query_token_i, query_pos_i = query_splits[:2]
    key_token_j, key_pos_j = key_splits[:2]

    query_token_i = query_token_i.unsqueeze(2).expand(-1, -1, seq_len, -1)
    query_pos_i = query_pos_i.unsqueeze(2).expand(-1, -1, seq_len, -1)
    key_token_j = key_token_j.unsqueeze(1).expand(-1, seq_len, -1, -1)
    key_pos_j = key_pos_j.unsqueeze(1).expand(-1, seq_len, -1, -1)

    # Stack the base tensor groups
    input_tensors = [
        tok_i,
        pos_i,
        query_token_i,
        query_pos_i,
        tok_j,
        pos_j,
        key_token_j,
        key_pos_j,
    ]

    if physchem is not None:
        query_physchem_i, key_physchem_j = query_splits[2], key_splits[2]
        query_physchem_i = query_physchem_i.unsqueeze(2).expand(-1, -1, seq_len, -1)
        key_physchem_j = key_physchem_j.unsqueeze(1).expand(-1, seq_len, -1, -1)
        physchem_expanded = physchem.unsqueeze(2).expand(-1, -1, seq_len, -1)

        input_tensors.extend([query_physchem_i, key_physchem_j, physchem_expanded])

    return torch.stack(input_tensors, dim=3)


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


def generate_text(model, prompt_text, max_len=128, temperature=1.0, top_k=None):
    """
    Genera testo in stile Inferno di Dante partendo da un prompt.

    Args:
        model: Il modello QuantumTransformerModel o Transformer_Model addestrato.
        prompt_text (str): Il testo iniziale da cui partire per la generazione.
        max_len (int): La lunghezza massima della sequenza generata.
        temperature (float): Fattore per riscalare i logits prima del softmax. Valori > 1 aumentano la casualità, < 1 la riducono.
        top_k (int, optional): Se specificato, considera solo i top_k token più probabili per il campionamento.

    Returns:
        str: Il testo generato.
    """
    model.eval()
    device = next(model.parameters()).device

    # Crea una lista dei caratteri unici nel dataset
    # Carica il testo di Dante per ottenere il vocabolario
    with open('./dataset/inferno.txt', 'r', encoding='utf-8') as f:
        dante_text = f.read()
    
    # Crea il vocabolario (caratteri unici nel testo + token speciali)
    chars = sorted(list(set(dante_text)))
    vocab = ["<pad>", "[CLS]", "[EOS]"] + chars
    
    # Crea mappature da caratteri a indici e viceversa
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    
    # Indici speciali
    cls_token = '[CLS]'
    eos_token = '[EOS]'
    cls_idx = stoi[cls_token]
    eos_idx = stoi[eos_token]

    # Tokenizza il prompt iniziale
    prompt_tokens = [cls_token] + [c for c in prompt_text if c in stoi]
    idx = torch.tensor([stoi[token] for token in prompt_tokens], dtype=torch.long, device=device).unsqueeze(0)

    # Loop di generazione
    with torch.no_grad():
        for _ in range(max_len - len(prompt_tokens)):
            # Controlla la dimensione della sequenza rispetto al block_size del modello
            block_size = model.position_embed.size(1)
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            
            # Ottieni i logits dal modello
            # Per la compatibilità con il modello condizionale, passiamo anche i valori zero per le proprietà fisico-chimiche
            physchem_props = torch.zeros(9, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(idx_cond, physchem_props)
            
            # Prendi l'ultimo token predetto
            logits = logits[:, -1, :]
            
            # Applica temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
                
            # Applica Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Calcola le probabilità e campiona
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Aggiungi il token alla sequenza
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Se viene generato [EOS], interrompi
            if idx_next.item() == eos_idx:
                break
                
    # Decodifica la sequenza generata
    generated_indices = idx[0].tolist()
    # Rimuovi i token speciali e mantieni solo i caratteri
    generated_text = ''.join([itos[i] for i in generated_indices 
                              if itos[i] not in ['[CLS]', '[EOS]', '<pad>']])
                              
    return generated_text