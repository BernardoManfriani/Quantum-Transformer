#!/usr/bin/env python
"""
Run script for Dante text generation model training.
This is a convenience wrapper around the main.py trainDante command.
"""

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Quantum Transformer on Dante's Inferno")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for sequence")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every n epochs")
    parser.add_argument("--fast", action="store_true", help="Run a fast training for testing")
    
    args = parser.parse_args()
    
    # Build command line arguments for main.py
    cmd_args = ["trainDante"]
    if args.epochs != 20:
        cmd_args.extend(["--epochs", str(args.epochs)])
    if args.batch_size != 16:
        cmd_args.extend(["--batch_size", str(args.batch_size)])
    if args.block_size != 128:
        cmd_args.extend(["--block_size", str(args.block_size)])
    if args.lr != 3e-4:
        cmd_args.extend(["--lr", str(args.lr)])
    if args.save_every != 5:
        cmd_args.extend(["--save_every", str(args.save_every)])
    if args.fast:
        cmd_args.append("--fast")
    
    # Inject arguments into sys.argv
    sys.argv = [sys.argv[0]] + cmd_args
    
    # Import and run main
    from main import main
    main()