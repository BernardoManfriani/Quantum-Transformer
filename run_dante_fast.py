  import torch
import os
import argparse
from train_text import train_dante_fast, generate_sample_text

def main():
    parser = argparse.ArgumentParser(description="Fast training on Inferno dataset")
    parser.add_argument("--data-path", type=str, default="./dataset/inferno_small.txt",
                       help="Path to the text dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="./dante_fast_checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-3,
                       help="Learning rate")
    parser.add_argument("--block-size", type=int, default=32,
                       help="Context window size")
    parser.add_argument("--embed-dim", type=int, default=16,
                       help="Embedding dimension")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"],
                       default="gpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--generate", action="store_true",
                       help="Generate text after training")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate text using the latest checkpoint")
    parser.add_argument("--gen-length", type=int, default=100,
                       help="Length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for text generation")
                       
    args = parser.parse_args()
    
    # Check for CUDA availability
    if args.device == "gpu" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Generate text only mode
    if args.generate_only:
        checkpoint_path = os.path.join(args.checkpoint_dir, "model_final.pt")
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}, cannot generate text")
            return
            
        print(f"Loading model from {checkpoint_path} for text generation")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model_config = checkpoint['training_configuration']
        
        # Load dataset to get vocabulary
        from src.transformer import Transformer_Dataset
        dataset = Transformer_Dataset(
            data_path=model_config.get("data_path", args.data_path),
            block_size=model_config.get("block_size", args.block_size)
        )
        
        # Initialize model with the same parameters
        from src.transformer import Transformer_Model
        model = Transformer_Model(
            qpu_count=1,
            vocab_size=len(dataset.vocab),
            embed_dim=model_config.get("embed_dim", args.embed_dim),
            block_size=model_config.get("block_size", args.block_size),
            classical_attention=True,
            num_qubits=model_config.get("num_qubits", 2),
            ansatz_layers=model_config.get("ansatz_layers", 1),
            conditional_training=False,
            classical_parameter_reduction=False,
        )
        
        # Load the model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to the right device
        device = torch.device("cuda:0" if args.device == "gpu" else "cpu")
        model.to(device)
        model.eval()
        
        # Generate text
        print("\n--- Generating text ---")
        print("Using temperature:", args.temperature)
        sample = generate_sample_text(
            model, 
            dataset, 
            device, 
            max_tokens=args.gen_length, 
            temperature=args.temperature
        )
        print("\nGenerated text:")
        print(sample)
        return
    
    # Train mode
    print("Starting fast training with the following parameters:")
    print(f"- Data path: {args.data_path}")
    print(f"- Checkpoint directory: {args.checkpoint_dir}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Block size: {args.block_size}")
    print(f"- Embedding dimension: {args.embed_dim}")
    print(f"- Device: {args.device}")
    print(f"- Seed: {args.seed}")
    
    # Train the model
    model, dataset = train_dante_fast(
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        block_size=args.block_size,
        embed_dim=args.embed_dim,
        device=args.device,
        seed=args.seed
    )
    
    # Generate more text if requested
    if args.generate:
        print("\n--- Generating more text ---")
        print("Using temperature:", args.temperature)
        device = torch.device("cuda:0" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
        sample = generate_sample_text(
            model, 
            dataset, 
            device, 
            max_tokens=args.gen_length, 
            temperature=args.temperature
        )
        print("\nGenerated text:")
        print(sample)

if __name__ == "__main__":
    main()