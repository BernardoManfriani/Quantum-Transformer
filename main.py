import torch
import argparse
from src.utils import generate_smiles, generate_text
from src.quantum_transformer import QuantumTransformerModel
from src.dante_trainer import train_dante_model
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 1
SEQ_LENGTH = 10 
VOCAB_SIZE = 33 
EMBED_DIM = 64  
BLOCK_SIZE = 24 
NUM_QUBITS = 6  
ANSATZ_LAYERS = 1
QPU_COUNT = 1

def main():
    parser = argparse.ArgumentParser(description="Quantum Transformer for text and molecule generation")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train Dante command
    train_parser = subparsers.add_parser("trainDante", help="Train the model on Dante's Inferno")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    train_parser.add_argument("--block_size", type=int, default=128, help="Block size for sequence")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every n epochs")
    train_parser.add_argument("--fast", action="store_true", help="Run a fast training for testing")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text or SMILES")
    generate_parser.add_argument("--model", choices=["smile", "dante"], default="smile", help="Model to use for generation")
    generate_parser.add_argument("--prompt", type=str, default="C" if "smile" else "Nel mezzo del cammin", help="Prompt for generation")
    generate_parser.add_argument("--max_len", type=int, default=100, help="Maximum length to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    generate_parser.add_argument("--top_k", type=int, default=5, help="Top-k for sampling")
    generate_parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to load")
    
    args = parser.parse_args()
    
    if args.command == "trainDante":
        train_dante_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            block_size=args.block_size,
            learning_rate=args.lr,
            save_every=args.save_every,
            fast_mode=args.fast
        )
        
    elif args.command == "generate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if args.model == "smile":
            vocab_size = 33
            block_size = BLOCK_SIZE
            checkpoint_path = args.checkpoint if args.checkpoint else './model_checkpoints/smile/model_epoch_20.pt'
        else:  # dante
            vocab_size = 79  # Updated from 78 to 79 based on the error
            block_size = 8  # Use the block size from the checkpoint
            checkpoint_path = args.checkpoint if args.checkpoint else './model_checkpoints/dante/best_dante_model.pt'
        
        # Make sure checkpoint exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            available_checkpoints = []
            if args.model == "dante":
                checkpoint_dir = './model_checkpoints/dante/'
                if os.path.exists(checkpoint_dir):
                    available_checkpoints = os.listdir(checkpoint_dir)
            if available_checkpoints:
                logger.info(f"Available checkpoints: {available_checkpoints}")
            return
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # First load the checkpoint to get the actual parameters
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Initialize model with the correct dimensions from the checkpoint
        model = QuantumTransformerModel(
            qpu_count=QPU_COUNT,
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            block_size=block_size,
            num_qubits=NUM_QUBITS,
            ansatz_layers=ANSATZ_LAYERS,
            quantum_gradient_method='spsa', 
            epsilon=0.01
        )
        
        # Load the state dict
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
     
        model.eval() 
        model.to(device)
        
        if args.model == "smile":
            generated_smiles = generate_smiles(
                model, 
                args.prompt,
                max_len=args.max_len,
                temperature=args.temperature,
                top_k=args.top_k
            )
            print(f"Generated SMILES: {generated_smiles}")
        else:  # dante
            prompt_tokens = ['[CLS]'] + list(args.prompt)
            generated_text = generate_text(
                model=model,
                prompt_tokens=prompt_tokens,
                max_len=args.max_len,
                temperature=args.temperature,
                top_k=args.top_k,
                block_size=block_size
            )
            print(f"Generated text: {generated_text}")
    
    else:
        # Default behavior if no command is provided (backward compatibility)
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
