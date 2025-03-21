import argparse

from src.analysis import (
    generate_plots_from_checkpoint,
    get_attention_maps,
    iterate_generations,
)
from src.train import train_transformer


def train_models(model_type, mode):
    """Function to train models."""
    print(f"Training {model_type} model with {mode} mode...")

    checkpoint_dir = f"./reproduced_checkpoints_{model_type}_{mode}"

    train_transformer(
        training_data="./dataset/qm9.csv",
        checkpoint_dir=checkpoint_dir,
        save_every_n_batches=10,
        attn_type="quantum" if model_type == "quantum" else "classical",
        conditional_training=True if mode == "conditions" else False,
        classical_parameter_reduction=True if model_type == "classical_eq" else False,
    )


def generate_figures():
    """Function to generate figures."""
    print("Generating figures from pre-trained models...")

    generate_plots_from_checkpoint(
        quantum_checkpoint_path="./model_checkpoints/quantum_sequence/model_epoch_20.pt",
        classical_checkpoint_path="./model_checkpoints/classical_sequence/model_epoch_20.pt",
        classical_equal_param_checkpoint_path="./model_checkpoints/classical_eq_sequence/model_epoch_20.pt",
        plot_train_losses=True,
        plot_val_losses=False,
        rolling_average=False,
        rolling_window=3,
        title=None,
        save_path="./reproduced_figures/sequence_only_training_losses.png",
        show_plot=False,
    )
    print("Saved ./reproduced_figures/sequence_only_training_losses.png")

    generate_plots_from_checkpoint(
        quantum_checkpoint_path="./model_checkpoints/quantum_sequence/model_epoch_20.pt",
        classical_checkpoint_path="./model_checkpoints/classical_sequence/model_epoch_20.pt",
        classical_equal_param_checkpoint_path="./model_checkpoints/classical_eq_sequence/model_epoch_20.pt",
        plot_train_losses=False,
        plot_val_losses=True,
        rolling_average=True,
        rolling_window=3,
        title=None,
        save_path="./reproduced_figures/sequence_only_validation_losses.png",
        show_plot=False,
    )
    print("Saved ./reproduced_figures/sequence_only_validation_losses.png")

    generate_plots_from_checkpoint(
        quantum_checkpoint_path="./model_checkpoints/quantum_conditions/model_epoch_20.pt",
        classical_checkpoint_path="./model_checkpoints/classical_conditions/model_epoch_20.pt",
        classical_equal_param_checkpoint_path="./model_checkpoints/classical_eq_conditions/model_epoch_20.pt",
        plot_train_losses=True,
        plot_val_losses=False,
        rolling_average=False,
        rolling_window=3,
        title=None,
        save_path="./reproduced_figures/conditions_training_losses.png",
        show_plot=False,
    )
    print("Saved ./reproduced_figures/conditions_training_losses.png")

    generate_plots_from_checkpoint(
        quantum_checkpoint_path="./model_checkpoints/quantum_conditions/model_epoch_20.pt",
        classical_checkpoint_path="./model_checkpoints/classical_conditions/model_epoch_20.pt",
        classical_equal_param_checkpoint_path="./model_checkpoints/classical_eq_conditions/model_epoch_20.pt",
        plot_train_losses=False,
        plot_val_losses=True,
        rolling_average=True,
        rolling_window=3,
        title=None,
        save_path="./reproduced_figures/conditions_validation_losses.png",
        show_plot=False,
    )
    print("Saved ./reproduced_figures/conditions_validation_losses.png")

    get_attention_maps(
        checkpoint_path="./model_checkpoints/quantum_conditions/model_epoch_20.pt",
        save_dir="./reproduced_figures/attention_maps_quantum_conditions",
        smiles_list=["O=[N+]([O-])c1ccoc1"],
    )
    print("Saved ./reproduced_figures/attention_maps_quantum_conditions")

    get_attention_maps(
        checkpoint_path="./model_checkpoints/classical_conditions/model_epoch_15.pt",
        save_dir="./reproduced_figures/attention_maps_classical_conditions",
        smiles_list=["O=[N+]([O-])c1ccoc1"],
    )
    print("Saved ./reproduced_figures/attention_maps_classical_conditions")

    get_attention_maps(
        checkpoint_path="./model_checkpoints/classical_eq_conditions/model_epoch_17.pt",
        save_dir="./reproduced_figures/attention_maps_classical_eq_conditions",
        smiles_list=["O=[N+]([O-])c1ccoc1"],
    )
    print("Saved ./reproduced_figures/attention_maps_classical_eq_conditions")


def run_inference(model_type, mode):
    """Function to run inference."""
    print(f"Running inference on {model_type} model with {mode} mode...")

    # select checkpoint path
    if model_type == "quantum" and mode == "sequence":
        checkpoint_path = "./model_checkpoints/quantum_sequence/model_epoch_17.pt"
    elif model_type == "classical" and mode == "sequence":
        checkpoint_path = "./model_checkpoints/classical_sequence/model_epoch_17.pt"
    elif model_type == "classical_eq" and mode == "sequence":
        checkpoint_path = "./model_checkpoints/classical_eq_sequence/model_epoch_16.pt"
    elif model_type == "quantum" and mode == "conditions":
        checkpoint_path = "./model_checkpoints/quantum_conditions/model_epoch_20.pt"
    elif model_type == "classical" and mode == "conditions":
        checkpoint_path = "./model_checkpoints/classical_conditions/model_epoch_15.pt"
    elif model_type == "classical_eq" and mode == "conditions":
        checkpoint_path = (
            "./model_checkpoints/classical_eq_conditions/model_epoch_17.pt"
        )

    iterate_generations(
        checkpoint_path=checkpoint_path,
        save_path=f"./reproduced_results/{model_type}_{mode}_results/",
        training_data_path="./dataset/train_dataset.csv",
    )

    print(
        f"Check ./reproduced_results/{model_type}_{mode}_results/OVERALL_SAMPLING_RESULTS.csv"
    )


def main():
    parser = argparse.ArgumentParser(description="Reproduce results from the paper")

    # Main flags
    parser.add_argument("--train-models", action="store_true", help="Train models")
    parser.add_argument("--figures", action="store_true", help="Generate figures")
    parser.add_argument(
        "--inference-results", action="store_true", help="Run inference"
    )

    # Sub-options for train-models and inference-results
    model_choices = ["quantum", "classical_eq", "classical"]
    mode_choices = ["sequence", "conditions"]

    parser.add_argument("--model", choices=model_choices, help="Specify the model type")
    parser.add_argument(
        "--mode", choices=mode_choices, help="Specify training or inference mode"
    )

    args = parser.parse_args()

    # Handling each flag
    if args.train_models:
        if not args.model or not args.mode:
            print("Error: --model and --mode must be specified with --train-models")
            return
        train_models(args.model, args.mode)

    if args.figures:
        generate_figures()

    if args.inference_results:
        if not args.model or not args.mode:
            print(
                "Error: --model and --mode must be specified with --inference-results"
            )
            return
        run_inference(args.model, args.mode)


if __name__ == "__main__":
    main()
