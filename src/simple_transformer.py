from torch import nn
from src.attenquantum_layer tion import AttentionQuantumLayer
class Simple_Quanttum_Transformer_Model(nn.Module):

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

        print("Intializing Simple Quantum Transformer Model...")
        # Initialize the model parameters
        if num_qubits % 2 != 0:
            raise ValueError(
                "The number of qubits must be even for the quantum circuit to work properly."
            )
        else:
            num_qubits_per_register = num_qubits // 2

        # Initialize quantum embeddings 
        # SI RIMUOVE LA PARTE DI SELF CLASSICAL ATTENTION
        print("Initializing quantum embeddings...")
        self._initialize_quantum_embeddings(
            vocab_size,
            ansatz_layers,
            num_qubits_per_register,
            block_size,
        )

        # Initialize classical embeddings
        print("Initializing classical embeddings...")
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))

        # Transformer architecture components
        print("Initializing transformer architecture components...")
        self.dropout = nn.Dropout(0.1)
        self.block = Simple_Quanttum_Transformer_Block(
            embed_dim=embed_dim,
            num_qubits=num_qubits,
            ansatz_layers=ansatz_layers,
            quantum_gradient_method=quantum_gradient_method,
            epsilon=epsilon,
            qpu_count=qpu_count,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    
    # Function to initialize the quantum embeddings
    def _initialize_quantum_embeddings(
        self,
        vocab_size,
        ansatz_layers,
        num_qubits_per_register,
        block_size,
    ):
        """Initializes quantum embeddings with the same dimensionality as the quantum model parameters. Parameteres are the angles of the quantum circuit."""

        self.token_embed_quantum_parameters = nn.Embedding(
            vocab_size, ansatz_layers * num_qubits_per_register
        )

        self._scale_quantum_parameters(self.token_embed_quantum_parameters.weight)
        
        self.position_embed_quantum_parameters = nn.Parameter(
            torch.zeros(1, block_size, ansatz_layers * num_qubits_per_register)
        )

    def _scale_quantum_parameters(self, tensor):
        """Scales quantum parameters for better Hilbert space coverage."""
        with torch.no_grad():
            min_val, max_val = tensor.min(), tensor.max()
            scaled_weights = (tensor - min_val) / (max_val - min_val) * np.pi
            tensor.copy_(scaled_weights)

    def forward(
        self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass through the model.
        """
        print("Forward...")
        # Get the batch size and sequence length
        B, T = idx.size()
        
        x = self.dropout(self.token_embed(idx) + self.position_embed[:, :T, :])

        angles = None

        position_embedding_angles = self.position_embed_quantum_parameters[
            :, :T, :
        ].expand(B, T, self.position_embed_quantum_parameters.shape[-1])

        angles = torch.stack(  
            [self.token_embed_quantum_parameters(idx), position_embedding_angles],
        )

        x, attn_weight = self.block(x, angles)
        x = self.layer_norm(x)
        logits = self.output(x)

        return logits, attn_weight

class Simple_Quantum_Transformer_Block(nn.Module):
    """"
    Simple Quantum Transformer Block with quantum attention
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
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Initialize attention mechanism
        self.attention = AttentionQuantumLayer(
            qpu_count=qpu_count,
            embed_dim=embed_dim,
            shift=torch.tensor(torch.pi / 2),
            ansatz_layers=ansatz_layers,
            num_qubits=num_qubits,
            quantum_gradient_method=quantum_gradient_method,
            epsilon=epsilon,
        )

        # Multi-layer perceptron -> feed forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x, angles, reduced_x):

        y, attn_weight = self.attention(self.layer_norm1(x), angles, reduced_x)
        x = x + y
        x = x + self.mlp(self.layer_norm2(x))

        return x, attn_weight


if __name__ == '__main__':

    # --- Parametri  ---
    BATCH_SIZE = 4
    SEQ_LENGTH = 10 # Deve essere <= block_size
    VOCAB_SIZE = 33 # Dimensione del vocabolario
    EMBED_DIM = 32  # Dimensione embedding classico (per V)
    BLOCK_SIZE = 20 # Max lunghezza sequenza
    NUM_QUBITS = 6  # Deve essere pari
    ANSATZ_LAYERS = 1
    QPU_COUNT = 1 # Esempio

    # --- Creazione Modello ---
    model = QuantumTransformerModel(
        qpu_count=QPU_COUNT,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        block_size=BLOCK_SIZE,
        num_qubits=NUM_QUBITS,
        ansatz_layers=ANSATZ_LAYERS,
        quantum_gradient_method='spsa', # o 'param-shift' se supportato da AttentionQuantumLayer
        epsilon=0.01
    )

    # --- Input ---
    vocab = ['#', '(', ')', '-', '1', '2', '3', '4', '5', '<pad>', '=', 'C', 'F', 'N', 'O', '[C-]', '[CH-]', '[CLS]', '[EOS]', '[N+]', '[N-]', '[NH+]', '[NH2+]', '[NH3+]', '[O-]', '[c-]', '[cH-]', '[n-]', '[nH+]', '[nH]', 'c', 'n', 'o']

    # Crea un batch di sequenze di indici (numeri interi < VOCAB_SIZE)
    input_indices = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    # print(f"\nInput indices shape: {input_indices.shape}")

    # --- Forward Pass ---
    try:
        logits, attention_weights = model(input_indices)
        print("\n--- Risultati Forward Pass ---")
        print(f"Logits shape: {logits.shape}") # Atteso: (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
        print(f"Attention weights shape: {attention_weights.shape}") # Atteso: (BATCH_SIZE, SEQ_LENGTH, SEQ_LENGTH) o simile, a seconda di AttentionQuantumLayer
    except ValueError as e:
         print(f"\nErrore durante il forward pass: {e}")
    except Exception as e:
        print(f"\nErrore generico durante il forward pass: {e}")




def tokenize_smiles(smiles_string, vocab):
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    pattern = '|'.join(re.escape(token) for token in sorted_vocab)
    regex = re.compile(pattern)
    tokens = regex.findall(smiles_string)
    if "".join(tokens) != smiles_string:
        print(f"Warning: tokenization may be incomplete for '{smiles_string}' -> {''.join(tokens)}")
    return tokens

test_smiles = "CC1C2C(=O)C3N2C13C"
tokenized_test = tokenize_smiles(test_smiles, vocab)
print(f"'{test_smiles}' -> {tokenized_test}")
