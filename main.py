import argparse
import os
import torch
from src.utils import generate_smiles, generate_text
from src.quantum_transformer import QuantumTransformerModel
from src.train_dante import train_model_dante

# Parametri di default
BATCH_SIZE = 16
SEQ_LENGTH = 64
VOCAB_SIZE_TEXT = 78  # Per il testo italiano
VOCAB_SIZE_SMILES = 33  # Per SMILES
EMBED_DIM = 64
BLOCK_SIZE = 128
NUM_QUBITS = 6
ANSATZ_LAYERS = 1
QPU_COUNT = 1
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
CHECKPOINT_DIR_DANTE = './model_checkpoints/dante'
CHECKPOINT_DIR_SMILES = './model_checkpoints/smiles'

def generate_dante(args):
    """Gestisce la generazione di testo dantesco"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Inizializza il modello per testo
    model = QuantumTransformerModel(
        qpu_count=args.qpu_count,
        vocab_size=VOCAB_SIZE_TEXT,
        embed_dim=args.embed_dim,
        block_size=args.block_size,
        num_qubits=args.num_qubits,
        ansatz_layers=args.ansatz_layers,
        quantum_gradient_method='spsa', 
        epsilon=0.01
    )
    
    # Carica il modello addestrato
    model.eval()
    model.to(device)
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Usa il checkpoint di default di Dante
        checkpoint_path = os.path.join(CHECKPOINT_DIR_DANTE, "best_dante_model.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"Caricamento checkpoint da {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Attenzione: checkpoint {checkpoint_path} non trovato. Utilizzo modello non addestrato.")
    
    # Genera il testo basato sul prompt
    prompt_tokens = ["[CLS]"] + list(args.prompt)
    generated_output = generate_text(
        model=model,
        prompt_tokens=prompt_tokens,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        block_size=args.block_size
    )
    print("\n--- Testo Generato ---")
    print(generated_output)
    
    return generated_output

def generate_smiles(args):
    """Gestisce la generazione di molecole SMILES"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    # Inizializza il modello per SMILES
    model = QuantumTransformerModel(
        qpu_count=args.qpu_count,
        vocab_size=VOCAB_SIZE_SMILES,
        embed_dim=args.embed_dim,
        block_size=args.block_size,
        num_qubits=args.num_qubits,
        ansatz_layers=args.ansatz_layers,
        quantum_gradient_method='spsa', 
        epsilon=0.01
    )
    
    # Carica il modello addestrato
    model.eval()
    model.to(device)
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Usa il checkpoint di default per SMILES
        checkpoint_path = os.path.join(CHECKPOINT_DIR_SMILES, "model_epoch_20.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"Caricamento checkpoint da {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Attenzione: checkpoint {checkpoint_path} non trovato. Utilizzo modello non addestrato.")
    
    # Genera SMILES basato sul prompt
    generated_output = generate_smiles(
        model=model,
        prompt_smiles=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print("\n--- SMILES Generato ---")
    print(generated_output)
    
    return generated_output

def train(args):
    """Gestisce l'addestramento del modello sul testo di Dante"""
    # Assicurati che esista la directory per i checkpoint di Dante
    os.makedirs(CHECKPOINT_DIR_DANTE, exist_ok=True)
    
    # Aggiorna il percorso dei checkpoint per l'addestramento
    args.checkpoint_dir = CHECKPOINT_DIR_DANTE
    
    train_model_dante(args)

def main():
    parser = argparse.ArgumentParser(description='Quantum-Transformer: Genera testo dantesco o molecole SMILES')
    
    # Comandi principali
    subparsers = parser.add_subparsers(dest='command', help='Comando da eseguire')
    
    # Parser per la generazione di testo dantesco
    dante_parser = subparsers.add_parser('generateDante', help='Genera testo in stile dantesco')
    dante_parser.add_argument('--prompt', type=str, default="Nel mezzo del cammin",
                          help='Prompt iniziale per la generazione')
    dante_parser.add_argument('--max_len', type=int, default=100,
                          help='Lunghezza massima dell\'output generato')
    dante_parser.add_argument('--temperature', type=float, default=0.7,
                          help='Temperatura per il sampling (0=deterministico)')
    dante_parser.add_argument('--top_k', type=int, default=40,
                          help='Top-k token per il sampling')
    dante_parser.add_argument('--checkpoint', type=str, default=None,
                          help='Percorso al checkpoint del modello da caricare')
    dante_parser.add_argument('--qpu_count', type=int, default=QPU_COUNT,
                          help='Numero di QPU da utilizzare')
    dante_parser.add_argument('--embed_dim', type=int, default=EMBED_DIM,
                          help='Dimensione embedding')
    dante_parser.add_argument('--block_size', type=int, default=BLOCK_SIZE,
                          help='Dimensione massima della sequenza')
    dante_parser.add_argument('--num_qubits', type=int, default=NUM_QUBITS,
                          help='Numero di qubits')
    dante_parser.add_argument('--ansatz_layers', type=int, default=ANSATZ_LAYERS,
                          help='Numero di strati ansatz')
    
    # Parser per la generazione di SMILES
    smiles_parser = subparsers.add_parser('generateSmiles', help='Genera molecole in formato SMILES')
    smiles_parser.add_argument('--prompt', type=str, default="C",
                            help='Prompt iniziale SMILES per la generazione')
    smiles_parser.add_argument('--max_len', type=int, default=24,
                            help='Lunghezza massima dell\'output generato')
    smiles_parser.add_argument('--temperature', type=float, default=0.8,
                            help='Temperatura per il sampling (0=deterministico)')
    smiles_parser.add_argument('--top_k', type=int, default=5,
                            help='Top-k token per il sampling')
    smiles_parser.add_argument('--checkpoint', type=str, default=None,
                            help='Percorso al checkpoint del modello da caricare')
    smiles_parser.add_argument('--qpu_count', type=int, default=QPU_COUNT,
                            help='Numero di QPU da utilizzare')
    smiles_parser.add_argument('--embed_dim', type=int, default=EMBED_DIM,
                            help='Dimensione embedding')
    smiles_parser.add_argument('--block_size', type=int, default=24,
                            help='Dimensione massima della sequenza per SMILES')
    smiles_parser.add_argument('--num_qubits', type=int, default=NUM_QUBITS,
                            help='Numero di qubits')
    smiles_parser.add_argument('--ansatz_layers', type=int, default=ANSATZ_LAYERS,
                            help='Numero di strati ansatz')
    
    # Parser per l'addestramento su testo di Dante
    train_parser = subparsers.add_parser('trainDante', help='Addestra il modello sul testo di Dante')
    train_parser.add_argument('--text_path', type=str, default='./data/inferno.txt',
                            help='Percorso al file di testo')
    train_parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                            help='Dimensione del batch')
    train_parser.add_argument('--block_size', type=int, default=BLOCK_SIZE,
                            help='Dimensione massima della sequenza')
    train_parser.add_argument('--embed_dim', type=int, default=EMBED_DIM,
                            help='Dimensione embedding')
    train_parser.add_argument('--num_qubits', type=int, default=NUM_QUBITS,
                            help='Numero di qubits')
    train_parser.add_argument('--ansatz_layers', type=int, default=ANSATZ_LAYERS,
                            help='Numero di strati ansatz')
    train_parser.add_argument('--qpu_count', type=int, default=QPU_COUNT,
                            help='Numero di QPU da utilizzare')
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                            help='Learning rate')
    train_parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                            help='Numero di epoche')
    train_parser.add_argument('--fast', action='store_true',
                            help='Modalità rapida per test (meno epoche/batch)')
    
    # Fast train comando (modalità rapida direttamente dal main)
    fast_parser = subparsers.add_parser('fastDante', help='Avvia addestramento rapido su Dante (per test)')
    
    args = parser.parse_args()
    
    # Crea le directory per i checkpoint se non esistono
    os.makedirs(CHECKPOINT_DIR_DANTE, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR_SMILES, exist_ok=True)
    
    # Se viene richiesto l'addestramento rapido
    if args.command == 'fastDante':
        fast_args = argparse.Namespace(
            text_path='./data/inferno.txt',
            batch_size=4,
            block_size=32,
            embed_dim=EMBED_DIM,
            num_qubits=NUM_QUBITS,
            ansatz_layers=ANSATZ_LAYERS,
            qpu_count=QPU_COUNT,
            lr=LEARNING_RATE,
            epochs=3,
            checkpoint_dir=CHECKPOINT_DIR_DANTE,
            fast=True
        )
        train(fast_args)
        return
    
    # Se non viene specificato un comando, mostra l'help
    if not args.command:
        parser.print_help()
        return
    
    # Esegue il comando specificato
    if args.command == 'generateDante':
        generate_dante(args)
    elif args.command == 'generateSmiles':
        generate_smiles(args)
    elif args.command == 'trainDante':
        train(args)

if __name__ == '__main__':
    main()
