import torch
from src.utils import generate_smiles, generate_text
from src.quantum_transformer import QuantumTransformerModel

BATCH_SIZE = 1
SEQ_LENGTH = 10 
VOCAB_SIZE = 33 
EMBED_DIM = 64  
BLOCK_SIZE = 24 
NUM_QUBITS = 6  
ANSATZ_LAYERS = 1
QPU_COUNT = 1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QuantumTransformerModel(
        qpu_count=QPU_COUNT,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        block_size=BLOCK_SIZE,
        num_qubits=NUM_QUBITS,
        ansatz_layers=ANSATZ_LAYERS,
        quantum_gradient_method='spsa', 
        epsilon=0.01
    )
 
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    generated_smiles = generate_smiles(
      model, 
      "C", 
      max_len=24, 
      temperature=0.8, 
      top_k=5
    )
    print(f"generated_smiles: {generated_smiles}")

    generated_text = generate_text(
        model=model,
        prompt_tokens="Nel mezzo del cammin", 
        max_len=100,
        temperature=0,
        top_k=5,
        block_size=1
    )
    print(f"generated_text: {generated_text}")

if __name__ == '__main__':
    main()
