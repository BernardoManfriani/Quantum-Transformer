import torch
from src.utils import generate_text
from src.quantum_transformer import QuantumTransformerModel
from src.transformer import Transformer_Dataset

# --- Parametri modello ---
BATCH_SIZE = 4
BLOCK_SIZE = 128       # Lunghezza massima della sequenza per il testo
EMBED_DIM = 64        # Dimensione embedding classico (per V)
NUM_QUBITS = 6        # Deve essere pari
ANSATZ_LAYERS = 1
QPU_COUNT = 1
EPSILON = 0.01

def main():
    # --- Setup device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")

    # --- Carica il dataset e ottieni la dimensione del vocabolario ---
    dataset = Transformer_Dataset(data_path="./dataset/inferno.txt", block_size=BLOCK_SIZE)
    VOCAB_SIZE = len(dataset.vocab)
    print(f"Dimensione del vocabolario: {VOCAB_SIZE}")

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

    # --- Prompt e generazione di testo ---
    prompt = "Nel mezzo del cammin di nostra vita"
    target_len = 200  # Lunghezza massima del testo generato

    print("\n--- Generazione Testo in Stile Inferno ---")
    print(f"Prompt: {prompt}")
    
    generated_text = generate_text(model, prompt, max_len=target_len, temperature=0.8, top_k=5)
    print(f"\nTesto generato:\n{prompt}{generated_text}")

if __name__ == '__main__':
    main()
