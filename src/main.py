import torch
from src.model import QuantumTransformerModel
from src.utils import generate_smiles

# --- Parametri modello ---
BATCH_SIZE = 4
SEQ_LENGTH = 10       # Deve essere <= BLOCK_SIZE
VOCAB_SIZE = 33       # Dimensione del vocabolario
EMBED_DIM = 64        # Dimensione embedding classico (per V)
BLOCK_SIZE = 24       # Lunghezza massima della sequenza
NUM_QUBITS = 6        # Deve essere pari
ANSATZ_LAYERS = 1
QPU_COUNT = 1
EPSILON = 0.01

# --- Vocabolario (ordinato) ---
VOCAB = [
    '#', '(', ')', '-', '1', '2', '3', '4', '5', '<pad>', '=', 'C', 'F', 'N', 'O',
    '[C-]', '[CH-]', '[CLS]', '[EOS]', '[N+]', '[N-]', '[NH+]', '[NH2+]', '[NH3+]',
    '[O-]', '[c-]', '[cH-]', '[n-]', '[nH+]', '[nH]', 'c', 'n', 'o'
]

def main():
    # --- Setup device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Inizializza modello ---
    model = QuantumTransformerModel(
        qpu_count=QPU_COUNT,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        block_size=BLOCK_SIZE,
        num_qubits=NUM_QUBITS,
        ansatz_layers=ANSATZ_LAYERS,
        quantum_gradient_method='spsa',  # o 'param-shift'
        epsilon=EPSILON
    )
    model.eval()
    model.to(device)

    # --- Prompt e generazione SMILES ---
    prompt = "C"
    target_len = BLOCK_SIZE

    print("\n--- Generazione Molecola ---")
    generated_smiles = generate_smiles(model, prompt, max_len=target_len, temperature=0.8, top_k=5)
    print(f"Prompt: {prompt}")
    print(f"Generato (max_len={target_len}): {generated_smiles}")

if __name__ == '__main__':
    main()
