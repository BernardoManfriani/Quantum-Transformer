import csv
import logging
import os
from typing import List, Optional, Tuple, Union

import cudaq
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem, RDLogger
from sklearn.experimental import enable_iterative_imputer  # noqa F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from torch.nn import functional as F
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from src.transformer import Transformer_Dataset, Transformer_Model
from src.utils import get_physchem_properties

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def generate_smiles(
    checkpoint_path: str,
    save_dir: str,
    choose_best_val_epoch: bool = True,
    num_of_model_queries: int = 100000,
    sampling_batch_size: int = 10000,
    MW: Union[float, np.float64] = np.nan,
    HBA: Union[float, np.float64] = np.nan,
    HBD: Union[float, np.float64] = np.nan,
    nRot: Union[float, np.float64] = np.nan,
    nRing: Union[float, np.float64] = np.nan,
    nHet: Union[float, np.float64] = np.nan,
    TPSA: Union[float, np.float64] = np.nan,
    LogP: Union[float, np.float64] = np.nan,
    StereoCenters: Union[float, np.float64] = np.nan,
    imputation_method: str = "knn",
    imputation_dataset_path: Optional[str] = None,
    dataset_novelty_check_path: Optional[str] = None,
    device: Union[str, torch.device] = "gpu",
    qpu_count: int = -1,
) -> Tuple[float, float, float]:
    """
    Generate SMILES strings using a trained Transformer model.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        save_dir (Optional[str]): Path to save the generated SMILES strings.
        choose_best_val_epoch (bool): Choose the best validation epoch for evaluation from any previous epoch.
        num_of_model_queries (int): Number of attempts to generate SMILES strings.
        sampling_batch_size (int): Batch size for sampling.
        MW (Union[float, np.float64]): Molecular weight for conditional sampling.
        HBA (Union[float, np.float64]): Hydrogen bond acceptors for conditional sampling.
        HBD (Union[float, np.float64]): Hydrogen bond donors for conditional sampling.
        nRot (Union[float, np.float64]): Number of rotatable bonds for conditional sampling.
        nRing (Union[float, np.float64]): Number of rings for conditional sampling.
        nHet (Union[float, np.float64]): Number of heteroatoms for conditional sampling.
        TPSA (Union[float, np.float64]): Topological polar surface area for conditional sampling.
        LogP (Union[float, np.float64]): LogP for conditional sampling.
        StereoCenters (Union[float, np.float64]): Number of stereocenters for conditional sampling.
        imputation_method (str): Imputation method for missing physicochemical properties.
        imputation_dataset_path (str): Path to the imputation dataset. Default will be set to the training dataset specified by the train_id from the checkpoint.
        dataset_novelty_check_path (str): Path to the dataset for novelty check. Default will be set to the training dataset specified by the train_id from the checkpoint.
        device (str): Device for training, either 'cpu' or 'gpu'.
        qpu_count (int): Number of GPUs to use (-1 = all available GPUs).

    Returns:
        Tuple[float, float, float]: Average novelty, uniqueness, and validity of the generated SMILES strings
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    imputation_dataset_path = (
        f"./training_splits/train_dataset_{checkpoint['training_configuration']['train_id']}.csv"
        if imputation_dataset_path is None
        else imputation_dataset_path
    )
    dataset_novelty_check_path = (
        f"./training_splits/train_dataset_{checkpoint['training_configuration']['train_id']}.csv"
        if dataset_novelty_check_path is None
        else dataset_novelty_check_path
    )
    if not os.path.exists(imputation_dataset_path):
        logger.error(f"Training data not found at {imputation_dataset_path}")
        raise FileNotFoundError(
            f"Training data not found at {imputation_dataset_path}. Was the split training data for this training moved?"
        )

    seed = checkpoint["training_configuration"]["seed"]

    def _configure_quantum_target(device: str | torch.device, qpu_count: int) -> int:
        """Configure the quantum target based on device availability."""
        if isinstance(device, torch.device):
            target = (
                "nvidia"
                if device.type == "cuda" and cudaq.has_target("nvidia")
                else "qpp-cpu"
            )
        else:
            target = (
                "nvidia"
                if device == "gpu" and cudaq.has_target("nvidia")
                else "qpp-cpu"
            )
        cudaq.set_target(target, option="mqpu,fp32")
        effective_qpu_count = (
            cudaq.get_target().num_qpus() if qpu_count == -1 else qpu_count
        )
        logger.info(
            f"Quantum target set to: {target} with QPU count: {effective_qpu_count}"
        )
        return effective_qpu_count

    qpu_count = _configure_quantum_target(device, qpu_count)

    # if device instance is torch.device do nothing
    if isinstance(device, torch.device):
        pass
    else:
        if device not in {"cpu", "gpu"}:
            raise ValueError("Device must be either 'cpu' or 'gpu'.")
        device = torch.device(
            "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
        )

    # Create a local random generator
    local_generator = torch.Generator(device=device)
    local_generator.manual_seed(seed)

    dataset = Transformer_Dataset(
        data_path=checkpoint["training_configuration"]["training_data"],
        block_size=(
            22
            if "qm9" in checkpoint["training_configuration"]["training_data"]
            else None
        ),
    )

    if choose_best_val_epoch:
        # Choose the best validation epoch
        best_val_loss_index = (
            checkpoint["val_losses"].index(min(checkpoint["val_losses"])) + 1
        )
        best_model_filename = f"model_epoch_{best_val_loss_index}.pt"
        selected_transformer_params = os.path.join(
            os.path.dirname(checkpoint_path), best_model_filename
        )
        logger.info(
            f"Generating SMILES using {best_model_filename} with val_loss {min(checkpoint['val_losses'])}"
        )
        checkpoint = torch.load(
            selected_transformer_params, map_location="cpu", weights_only=False
        )

    else:
        logger.info(
            f"Generating SMILES using {os.path.basename(checkpoint_path)} with val_loss {checkpoint['val_losses'][-1]}"
        )

    model = Transformer_Model(
        vocab_size=len(dataset.vocab),
        embed_dim=64,
        block_size=dataset.block_size,
        classical_attention=checkpoint["training_configuration"]["classical_attention"],
        num_qubits=checkpoint["training_configuration"]["num_qubits"],
        ansatz_layers=checkpoint["training_configuration"]["ansatz_layers"],
        conditional_training=checkpoint["training_configuration"][
            "conditional_training"
        ],
        classical_parameter_reduction=checkpoint["training_configuration"][
            "classical_parameter_reduction"
        ],
        qpu_count=qpu_count,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    # Initialize counters for tracking validity and uniqueness
    total_generated = 0
    valid_molecules = 0

    # Set to store unique SMILES strings
    generated_smiles = []
    unique_smiles = set()

    sampling_properties_tensor = None
    required_properties = [
        "MW",
        "HBA",
        "HBD",
        "nRot",
        "nRing",
        "nHet",
        "TPSA",
        "LogP",
        "StereoCenters",
    ]
    sampling_conditions = [MW, HBA, HBD, nRot, nRing, nHet, TPSA, LogP, StereoCenters]

    if (
        any(
            not np.isnan(item) if isinstance(item, (float, int)) else True
            for item in sampling_conditions
        )
        and not checkpoint["training_configuration"]["conditional_training"]
    ):
        logger.warning(
            "Some physicochemical properties have been specified while model was trained without conditions. These properties will be ignored."
        )

    if bool(checkpoint["training_configuration"]["conditional_training"]):

        if any(np.isnan(x) for x in sampling_conditions):
            logger.debug(
                "Some physicochemical properties are missing. Imputation will be performed. It is recommended to use the training dataset for imputation. The './train_dataset.csv' file will be used by default. Specify the impuation dataset using the 'imputation_dataset' argument."
            )

            # Load training dataset
            imputation_data = pd.read_csv(imputation_dataset_path)

            # Check if all headers are present
            all_properties_present = set(required_properties).issubset(
                imputation_data.columns
            )

            if not all_properties_present:
                logger.debug(
                    f"Missing required properties in the imputation dataset. Required properties: {required_properties}. Computing properties manually."
                )

                # Calculate physico-chemical properties of the dataset if they are not provided
                imputation_dataset_properties_lists = []
                # Extract SMILES strings and compute properties

                # Determine which SMILES column to use
                smiles_column = (
                    "SMILES"
                    if "SMILES" in imputation_data.columns
                    else "smiles" if "smiles" in imputation_data.columns else None
                )

                if smiles_column:
                    for smiles in imputation_data[smiles_column]:
                        properties = get_physchem_properties(smiles)
                        imputation_dataset_properties_lists.append(properties)
                else:
                    logger.error(
                        "Missing SMILES strings in the imputation dataset. Please provide SMILES strings in a 'SMILES' or 'smiles' column."
                    )
                    assert bool(smiles_column), "Missing SMILES strings."

                # Convert the list of properties to a numpy array for easier computation
                imputation_dataset_properties = np.array(
                    imputation_dataset_properties_lists
                )  # Shape will be (num_samples, 9)
            else:
                imputation_dataset_properties = imputation_data[
                    required_properties
                ].to_numpy()

            logger.info(
                f"Imputing missing properties using the '{imputation_method}' method."
            )
            if imputation_method == "mean":
                imp = SimpleImputer(missing_values=np.nan, strategy="mean")
            elif imputation_method == "median":
                imp = SimpleImputer(missing_values=np.nan, strategy="median")
            elif imputation_method == "most_frequent":
                imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            elif imputation_method == "multivariate":
                imp = IterativeImputer(max_iter=10, random_state=seed)
            elif imputation_method == "knn":
                imp = KNNImputer(n_neighbors=100, weights="uniform")
            else:
                raise ValueError(
                    f"Invalid imputation method: '{imputation_method}'. "
                    "Please choose one of the following: 'mean', 'median', "
                    "'most_frequent', 'multivariate', or 'knn'. See https://scikit-learn.org/1.5/modules/impute.html for details."
                )

            imp.fit(imputation_dataset_properties)
            imputed_properties = imp.transform(
                np.array(sampling_conditions).reshape(1, -1)
            )
            sampling_properties_tensor = torch.tensor(
                imputed_properties[0], dtype=torch.float32
            ).to(device=device)

        else:
            sampling_properties_tensor = torch.tensor(
                sampling_conditions, dtype=torch.float32
            ).to(device=device)

    # Silence RDKit warnings
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    logger.debug("Sampling molecules...")

    sampled_molecular_properties = []
    # Sample until we reach the requested number of molecules
    with tqdm(
        total=num_of_model_queries, desc="Sampling molecules", leave=True
    ) as pbar:
        while total_generated < num_of_model_queries:

            # Adjust batch size if nearing the total number of molecules to generate
            batch_size_curr = min(
                sampling_batch_size, num_of_model_queries - total_generated
            )

            # Initialize sequences and tracking variables for the batch
            sequences = [["[CLS]"] for _ in range(batch_size_curr)]
            done = [False] * batch_size_curr
            lengths = [1] * batch_size_curr  # All sequences start with length 1

            # Continue generating tokens until all sequences in the batch are done
            while not all(done):

                # Convert sequences to indices
                token_idxs = [
                    torch.tensor([dataset.stoi[s] for s in seq], dtype=torch.long)
                    for seq in sequences
                ]

                # Pad sequences to the same length
                token_idxs_padded = torch.nn.utils.rnn.pad_sequence(
                    token_idxs, batch_first=True, padding_value=dataset.stoi["<pad>"]
                ).to(device)

                lengths = [len(seq) for seq in sequences]
                lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(device)

                # Generate logits from the model
                with torch.no_grad():
                    if bool(
                        checkpoint["training_configuration"]["conditional_training"]
                    ):
                        if sampling_properties_tensor is None:
                            raise ValueError(
                                "sampling_properties_tensor is None but required for conditional training."
                            )
                        logits, _ = model(
                            token_idxs_padded,
                            sampling_properties_tensor.expand(batch_size_curr, -1),
                        )  # Shape: (batch_size, seq_length, vocab_size)
                    else:
                        logits, _ = model(token_idxs_padded)

                # Get logits corresponding to the last token in each sequence
                logits_last = logits[
                    torch.arange(len(sequences)), lengths_tensor - 1, :
                ]  # Shape: (batch_size, vocab_size)

                # Compute probabilities and sample the next token
                probs = F.softmax(logits_last, dim=-1)
                next_tokens = torch.multinomial(
                    probs, num_samples=1, generator=local_generator
                ).squeeze(
                    -1
                )  # Shape: (batch_size)

                next_tokens_strings = [dataset.itos[idx.item()] for idx in next_tokens]

                # Update sequences and done flags
                for i in range(batch_size_curr):
                    if not done[i]:
                        next_token_str = next_tokens_strings[i]
                        sequences[i].append(next_token_str)
                        if (
                            next_token_str == "[EOS]"
                            or len(sequences[i]) >= dataset.block_size
                        ):
                            done[i] = True

            # Process the completed sequences
            for seq in sequences:
                total_generated += 1  # Increment the total molecules sampled

                try:
                    # Convert the generated sequence into SMILES and canonicalize it
                    smiles = (
                        "".join(seq)
                        .replace("[CLS]", "")
                        .replace("[EOS]", "")
                        .replace("<pad>", "")
                    )
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        canonical = Chem.MolToSmiles(mol, canonical=True)
                        # Check if the molecule is valid
                        if canonical:
                            # Calculate properties to validate further as some pass canonicalization but fail getting properties
                            properties = get_physchem_properties(canonical)
                            if (
                                properties
                            ):  # Only add if properties calculation is successful
                                valid_molecules += 1  # Increment valid molecules count
                                generated_smiles.append(canonical)
                                sampled_molecular_properties.append(properties)
                                unique_smiles.add(canonical)

                except Exception:
                    continue  # Continue to the next sample if there's an error

            pbar.update(batch_size_curr)  # Update the progress bar

    # Calculate properties for each SMILES and create a DataFrame
    df = pd.DataFrame(generated_smiles, columns=["SMILES"])
    properties_df = pd.DataFrame(
        sampled_molecular_properties, columns=required_properties
    )

    # Concatenate the DataFrames
    df = pd.concat([df, properties_df], axis=1)

    # Save the DataFrame to a CSV file
    df.to_csv(save_dir, index=False)
    num_valid_molecules = len(df)
    logger.info(
        f"{num_valid_molecules} valid molecules generated ({num_valid_molecules*100 / num_of_model_queries if num_of_model_queries > 0 else 0:.2f} % of sampled molecules)."
    )
    df.drop_duplicates(inplace=True)
    num_unique_molecules = len(df)
    logger.info(
        f"{num_unique_molecules} unique molecules generated ({num_unique_molecules*100 / num_valid_molecules if num_valid_molecules > 0 else 0:.2f} % of valid molecules)"
    )

    df.to_csv(save_dir.replace(".csv", "_unique.csv"), index=False)

    novelty_percentage = 0.0

    if dataset_novelty_check_path is not None:
        # Load the CSV into a DataFrame
        df_novelty_check = pd.read_csv(dataset_novelty_check_path)
        smiles_column = (
            "SMILES"
            if "SMILES" in df_novelty_check.columns
            else "smiles" if "smiles" in df_novelty_check.columns else None
        )
        if smiles_column is None:
            logger.error(
                "SMILES column not found in the dataset. Please ensure the dataset contains a 'SMILES' or 'smiles' column."
            )
            raise ValueError("SMILES not found.")
        novel_smiles = [
            smiles
            for smiles in unique_smiles
            if smiles not in df_novelty_check[smiles_column].values
        ]
        # Calculate properties of novel_smiles so they can be saved to csv
        novel_smiles_properties = []
        for smiles in novel_smiles:
            properties = get_physchem_properties(smiles)
            if properties:
                novel_smiles_properties.append(properties)

        novel_save_path = save_dir.replace(".csv", "_novel.csv")
        # save smiles and properties to csv
        novel_df = pd.DataFrame(novel_smiles, columns=["SMILES"])
        novel_properties_df = pd.DataFrame(
            novel_smiles_properties, columns=required_properties
        )
        novel_df = pd.concat([novel_df, novel_properties_df], axis=1)
        novel_df.to_csv(novel_save_path, index=False)

        novelty_percentage = (
            (len(novel_smiles) / len(unique_smiles)) * 100
            if len(unique_smiles) > 0
            else 0
        )
        logger.info(
            f"{len(novel_smiles)} novel molecules generated ({novelty_percentage:.2f} % of unique molecules)"
        )

    # Calculate percentages for validity and uniqueness
    validity_percentage = (
        (valid_molecules / num_of_model_queries) * 100
        if num_of_model_queries > 0
        else 0
    )
    uniqueness_percentage = (
        (len(unique_smiles) / valid_molecules) * 100 if valid_molecules > 0 else 0
    )

    return validity_percentage, uniqueness_percentage, novelty_percentage


def evaluate_accuracy(
    checkpoint_path: str,
    evaluation_batch_size: int = 32,
    choose_best_val_epoch: bool = False,
    device: str = "gpu",
    qpu_count: int = -1,
) -> float:
    """
    Evaluate the accuracy of the Transformer model on the validation dataset.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        evaluation_batch_size (int): Batch size for evaluation.
        choose_best_val_epoch (bool): Choose the best validation epoch for evaluation from any previous epoch.
        device (str): Device for training, either 'cpu' or 'gpu'.
        qpu_count (int): Number of GPUs to use (-1 = all available GPUs).

    Returns:
        float: Model accuracy on the validation dataset.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    seed = checkpoint["training_configuration"]["seed"]

    def _configure_quantum_target(
        device: Union[str, torch.device], qpu_count: int
    ) -> int:
        """Configure the quantum target based on device availability."""
        if isinstance(device, torch.device):
            target = (
                "nvidia"
                if device.type == "cuda" and cudaq.has_target("nvidia")
                else "qpp-cpu"
            )
        else:
            target = (
                "nvidia"
                if device == "gpu" and cudaq.has_target("nvidia")
                else "qpp-cpu"
            )
        cudaq.set_target(target, option="mqpu,fp32")
        effective_qpu_count = (
            cudaq.get_target().num_qpus() if qpu_count == -1 else qpu_count
        )
        logger.debug(
            f"Quantum target set to: {target} with QPU count: {effective_qpu_count}"
        )
        return effective_qpu_count

    qpu_count = _configure_quantum_target(device, qpu_count)

    if device not in {"cpu", "gpu"}:
        raise ValueError("Device must be either 'cpu' or 'gpu'.")
    torch_device: torch.device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )

    # Create a local random generator
    local_generator = torch.Generator(device=torch_device)
    local_generator.manual_seed(seed)

    # Load the dataset
    full_dataset = Transformer_Dataset(
        data_path=checkpoint["training_configuration"]["training_data"],
        block_size=(
            22
            if "qm9" in checkpoint["training_configuration"]["training_data"]
            else None
        ),
    )

    validation_data = f"./validation_splits/val_dataset_{checkpoint['training_configuration']['train_id']}.csv"

    validation_dataset = Transformer_Dataset(
        data_path=validation_data, block_size=22 if "qm9" in validation_data else None
    )

    # Validation data loader
    val_loader = StatefulDataLoader(
        validation_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=evaluation_batch_size,
        num_workers=0,
    )

    if choose_best_val_epoch:
        # Choose the best validation epoch
        best_val_loss_index = (
            checkpoint["val_losses"].index(min(checkpoint["val_losses"])) + 1
        )
        best_model_filename = f"model_epoch_{best_val_loss_index}.pt"
        selected_transformer_params = os.path.join(
            os.path.dirname(checkpoint_path), best_model_filename
        )
        logger.info(
            f"Evaluating Accuracy using {best_model_filename} with val_loss {min(checkpoint['val_losses'])}"
        )
        checkpoint = torch.load(
            selected_transformer_params, map_location="cpu", weights_only=False
        )

    else:
        logger.info(
            f"Evaluating Accuracy using {os.path.basename(checkpoint_path)} with val_loss {checkpoint['val_losses'][-1]}"
        )

    # Initialize model
    model = Transformer_Model(
        vocab_size=len(full_dataset.vocab),
        embed_dim=64,
        block_size=full_dataset.block_size,
        classical_attention=checkpoint["training_configuration"]["classical_attention"],
        num_qubits=checkpoint["training_configuration"]["num_qubits"],
        ansatz_layers=checkpoint["training_configuration"]["ansatz_layers"],
        conditional_training=checkpoint["training_configuration"][
            "conditional_training"
        ],
        classical_parameter_reduction=checkpoint["training_configuration"][
            "classical_parameter_reduction"
        ],
        qpu_count=qpu_count,
    ).to(torch_device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        val_pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc="Evaluating Accuracy",
        )
        for batch_id, batch in val_pbar:
            x, y, physchem_props = [item.to(torch_device) for item in batch]

            # Predict logits
            logits, _ = model(x, physchem_props)

            # Get predicted token (highest probability)
            predicted_tokens = logits.argmax(dim=-1)

            # Flatten predictions and ground truth
            y_flat = y.view(-1)
            predicted_flat = predicted_tokens.view(-1)

            # Mask padding tokens (-1) in ground truth
            valid_mask = y_flat != -1
            y_flat = y_flat[valid_mask]
            predicted_flat = predicted_flat[valid_mask]

            # Count correct predictions
            correct += (predicted_flat == y_flat).sum().item()
            total += valid_mask.sum().item()

            val_pbar.set_postfix(correct=correct, total=total, accuracy=correct / total)

    # Return overall accuracy
    return correct * 100 / total


def get_attention_maps(
    checkpoint_path: str,
    save_dir: str,
    smiles_list: Union[str, List[str]],
    choose_best_val_epoch: bool = True,
    show_plots: bool = False,
    device: str = "gpu",
    qpu_count: int = -1,
) -> None:
    """
    Generate attention maps for the given SMILES strings.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        save_path (Optional[str]): Directory to save the attention maps.
        smiles_list (List[str]): List of SMILES strings.
        choose_best_val_epoch (bool): Choose the best validation epoch for evaluation from any previous epoch.
        device (str): Device for training, either 'cpu' or 'gpu'.
        qpu_count (int): Number of GPUs to use (-1 = all available GPUs).
        show_plot (bool): Show the plot.

    Returns:
        None
    """

    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    if not isinstance(smiles_list, list):
        raise ValueError("smiles_list must be a string or a list of strings.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def _configure_quantum_target(
        device: Union[str, torch.device], qpu_count: int
    ) -> int:
        """Configure the quantum target based on device availability."""
        if isinstance(device, torch.device):
            target = (
                "nvidia"
                if device.type == "cuda" and cudaq.has_target("nvidia")
                else "qpp-cpu"
            )
        else:
            target = (
                "nvidia"
                if device == "gpu" and cudaq.has_target("nvidia")
                else "qpp-cpu"
            )
        cudaq.set_target(target, option="mqpu,fp32")
        effective_qpu_count = (
            cudaq.get_target().num_qpus() if qpu_count == -1 else qpu_count
        )
        logger.info(
            f"Quantum target set to: {target} with QPU count: {effective_qpu_count}"
        )
        return effective_qpu_count

    qpu_count = _configure_quantum_target(device, qpu_count)

    if device not in {"cpu", "gpu"}:
        raise ValueError("Device must be either 'cpu' or 'gpu'.")
    torch_device: torch.device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )

    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    dataset = Transformer_Dataset(
        data_path=checkpoint["training_configuration"]["training_data"],
        block_size=(
            22
            if "qm9" in checkpoint["training_configuration"]["training_data"]
            else None
        ),
    )

    # Build/Load model
    model = Transformer_Model(
        vocab_size=len(dataset.vocab),
        embed_dim=64,
        block_size=dataset.block_size,
        classical_attention=checkpoint["training_configuration"]["classical_attention"],
        num_qubits=checkpoint["training_configuration"]["num_qubits"],
        ansatz_layers=checkpoint["training_configuration"]["ansatz_layers"],
        conditional_training=checkpoint["training_configuration"][
            "conditional_training"
        ],
        classical_parameter_reduction=checkpoint["training_configuration"][
            "classical_parameter_reduction"
        ],
        qpu_count=qpu_count,
    ).to(torch_device)

    if choose_best_val_epoch:
        # Choose the best validation epoch
        best_val_loss_index = (
            checkpoint["val_losses"].index(min(checkpoint["val_losses"])) + 1
        )
        best_model_filename = f"model_epoch_{best_val_loss_index}.pt"
        selected_transformer_params = os.path.join(
            os.path.dirname(checkpoint_path), best_model_filename
        )
        logger.info(
            f"Using {best_model_filename} with val_loss {min(checkpoint['val_losses'])}"
        )
        checkpoint = torch.load(
            selected_transformer_params, map_location="cpu", weights_only=False
        )
    else:
        logger.info(
            f"Using {os.path.basename(checkpoint_path)} with val_loss {checkpoint['val_losses'][-1]}"
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    excluded_token_strings = ["[CLS]", "[EOS]", "<pad>"]

    for smiles in smiles_list:
        try:
            physchem_props = get_physchem_properties(smiles)
            input_sequence = dataset.tokenize_smiles(smiles)
            tokens = [dataset.itos[idx] for idx in input_sequence]
            input_tensor = (
                torch.tensor(input_sequence, dtype=torch.long)
                .unsqueeze(0)
                .to(torch_device)
            )

            # (debug) Print tokens to verify what they actually are
            logger.debug(f"Tokens for {smiles}: {tokens}")

            # Forward pass
            with torch.no_grad():
                _, attention_maps = model(
                    input_tensor, torch.tensor([physchem_props], device=torch_device)
                )

            if checkpoint["training_configuration"]["classical_attention"]:
                # Single-head classical attention: shape => (batch, head, T, T)
                # We'll take the first head: attention_maps[0, 0]
                attention_map = attention_maps[0, 0].detach().cpu().numpy()
            else:
                attention_map = attention_maps[0].detach().cpu().numpy()

            # Build mask to exclude special tokens
            valid_mask: NDArray[np.bool_] = np.array(
                [t not in excluded_token_strings for t in tokens], dtype=bool
            )

            # Filter out rows/columns of excluded tokens
            filtered_tokens = np.array(tokens)[valid_mask]
            filtered_attention_map = attention_map[valid_mask][:, valid_mask]

            # Plot with a white->blue colormap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                filtered_attention_map,
                cmap="Blues",
                square=True,
                cbar=True,
                xticklabels=filtered_tokens,
                yticklabels=filtered_tokens,
                vmin=0.0,
                vmax=1.0,
            )

            plt.xlabel("Tokens")
            plt.ylabel("Tokens")
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save figure
            output_path = os.path.join(
                save_dir, f"{smiles.replace('/', '_')}_attention.png"
            )

            # prepend the name of the checkpoint_path's directory to the output_path
            parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
            output_dir, output_filename = os.path.dirname(
                output_path
            ), os.path.basename(output_path)

            output_path = os.path.join(output_dir, f"{parent_dir}_{output_filename}")

            plt.savefig(output_path)
            plt.show() if show_plots else plt.close()

            logger.info(f"Saved attention map for {smiles} at {output_path}")

        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")


def generate_plots_from_checkpoint(
    quantum_checkpoint_path: Optional[str] = None,
    classical_checkpoint_path: Optional[str] = None,
    classical_equal_param_checkpoint_path: Optional[str] = None,
    plot_train_losses: bool = True,
    plot_val_losses: bool = False,
    rolling_average: bool = False,
    rolling_window: int = 3,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Generate plots for manuscript based on PyTorch checkpoints.

    Args:
        quantum_checkpoint_path (str): Path to the quantum checkpoint.
        classical_checkpoint_path (str): Path to the classical checkpoint.
        classical_equal_param_checkpoint_path (str): Path to the classical checkpoint with equal parameters.
        plot_train_losses (bool): Plot training losses.
        plot_val_losses (bool): Plot validation losses.
        rolling_average (bool): Plot rolling average.
        rolling_window (int): Rolling window size.
        title (str): Title for the plot.
        save_path (str): Path to save the plot.

    Returns:
        None
    """
    if (
        quantum_checkpoint_path is None
        and classical_checkpoint_path is None
        and classical_equal_param_checkpoint_path is None
    ):
        return
    if not plot_train_losses and not plot_val_losses:
        return

    # Load checkpoints if provided
    quantum_checkpoint = (
        torch.load(quantum_checkpoint_path, weights_only=False)
        if quantum_checkpoint_path
        else None
    )
    classical_checkpoint = (
        torch.load(classical_checkpoint_path, weights_only=False)
        if classical_checkpoint_path
        else None
    )
    classical_equal_param_checkpoint = (
        torch.load(classical_equal_param_checkpoint_path, weights_only=False)
        if classical_equal_param_checkpoint_path
        else None
    )

    num_quantum_epochs = 0
    num_classical_epochs = 0
    num_equal_param_epochs = 0

    # Find max number of epochs
    if quantum_checkpoint:
        num_quantum_epochs = len(quantum_checkpoint["training_losses"])
    if classical_checkpoint:
        num_classical_epochs = len(classical_checkpoint["training_losses"])
    if classical_equal_param_checkpoint:
        num_equal_param_epochs = len(
            classical_equal_param_checkpoint["training_losses"]
        )

    num_epochs = max(num_quantum_epochs, num_classical_epochs, num_equal_param_epochs)

    epochs_loss = np.arange(1, num_epochs + 1)

    def _rolling_average_curve(data, window=3):
        return (
            pd.Series(data)
            .rolling(window=window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )

    # Define colors
    quantum_colors = {"train": "blue", "val": "cyan"}
    classical_colors = {"train": "orange", "val": "gold"}
    equal_param_colors = {"train": "black", "val": "gray"}

    fig, ax1 = plt.subplots(figsize=(10, 5))

    lines = []
    labels = []

    # Helper to plot loss
    def plot_loss(ax, checkpoint, label_prefix, colors):
        if checkpoint is None or epochs_loss is None:
            return
        # Training losses
        if plot_train_losses and "training_losses" in checkpoint:
            data = checkpoint["training_losses"][:num_epochs]
            if rolling_average:
                smoothed = _rolling_average_curve(data, window=rolling_window)
                (line,) = ax.plot(
                    epochs_loss,
                    smoothed,
                    label=f"{label_prefix} Training Loss",
                    color=colors["train"],
                )
                ax.scatter(epochs_loss, data, color=colors["train"], alpha=1.0)
            else:
                (line,) = ax.plot(
                    epochs_loss,
                    data,
                    label=f"{label_prefix} Training Loss",
                    color=colors["train"],
                    alpha=0.3,
                    marker="o",
                )
            lines.append(line)
            labels.append(line.get_label())

        # Validation losses
        if plot_val_losses and "val_losses" in checkpoint:
            data = checkpoint["val_losses"][:num_epochs]
            if rolling_average:
                smoothed = _rolling_average_curve(data, window=rolling_window)
                (line,) = ax.plot(
                    epochs_loss,
                    smoothed,
                    label=f"{label_prefix} Validation Loss",
                    color=colors["val"],
                )
                ax.scatter(epochs_loss, data, color=colors["val"], alpha=1.0)
            else:
                (line,) = ax.plot(
                    epochs_loss,
                    data,
                    label=f"{label_prefix} Validation Loss",
                    color=colors["val"],
                    alpha=0.3,
                    marker="o",
                )
            lines.append(line)
            labels.append(line.get_label())

    plot_loss(ax1, quantum_checkpoint, "Quantum", quantum_colors)
    plot_loss(ax1, classical_checkpoint, "Classical", classical_colors)
    plot_loss(ax1, classical_equal_param_checkpoint, "Classical-eq", equal_param_colors)

    ax1.set_xlabel("Epochs", fontsize=16, weight="bold")
    ax1.set_ylabel("Loss", fontsize=16, weight="bold")
    ax1.set_xlim(1, num_epochs)
    ax1.set_xticks(range(1, num_epochs + 1))  # 1 to 20 inclusive
    ax1.tick_params(axis="y", labelsize=12)

    # Legend top right for only losses
    ax1.legend(loc="upper right", fontsize=20, frameon=False)

    if title:
        plt.title(title, fontsize=18, weight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show() if show_plot else plt.close()


def iterate_generations(
    checkpoint_path: str,
    save_path: str,
    training_data_path: Optional[str] = None,
    choose_best_val_epoch: bool = True,
    property_variation_factor: float = 2,
    central_tendency: str = "mean",
    variation: str = "std",
    imputation_method: str = "knn",
    num_samples: int = 100000,
    sampling_batch_size: int = 25000,
    device: str = "gpu",
    qpu_count: int = -1,
) -> None:
    """
    Sample molecules according to different property conditions. This function will iteratively call the `generate_smiles` function to sample molecules based on properties within the training distribution and outside the training distribution by central_tendency +/- property_variation_factor * variation. This function saves the generated SMILES strings to CSV files, as well as a summary CSV file with the average validity, uniqueness, novelty, and property values of the generated SMILES strings. The summary of results is saved to "OVERALL_SAMPLING_RESULTS.csv".

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        save_path (str): Path to save the generated SMILES strings.
        training_data_path (Optional[str]): Path to the training data CSV file. Used for property imputation and novelty checks. If not provided, the training data from the checkpoint train_id will be used.
        choose_best_val_epoch (bool): Choose the best validation epoch for evaluation for any previous epoch.
        property_variation_factor (float): Factor to vary the properties.
        central_tendency (str): Central tendency for property variation, either 'mean' or 'median'.
        variation (str): Variation method for property variation, either 'std' or 'iqr'.
        imputation_method (str): Imputation method for missing physicochemical properties.
        num_samples (int): Number of samples to generate.
        sampling_batch_size (int): Batch size for sampling.
        device (str): Device for training, either 'cpu' or 'gpu'.
        qpu_count (int): Number of GPUs to use (-1 = all available GPUs).

    Returns:
        None
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if training_data_path is None:
        specified_training_data = f"./training_splits/train_dataset_{checkpoint['training_configuration']['train_id']}.csv"

        if not os.path.exists(specified_training_data):
            logger.error(f"Training data not found at {specified_training_data}")
            raise FileNotFoundError(
                f"Training data not found at {specified_training_data}. Was the split training data for this training moved?"
            )
    else:
        specified_training_data = training_data_path

    def _configure_quantum_target(
        device: Union[str, torch.device], qpu_count: int
    ) -> int:
        """Configure the quantum target based on device availability."""
        if isinstance(device, torch.device):
            target = (
                "nvidia"
                if device.type == "cuda" and cudaq.has_target("nvidia")
                else "qpp-cpu"
            )
        else:
            target = (
                "nvidia"
                if device == "gpu" and cudaq.has_target("nvidia")
                else "qpp-cpu"
            )
        cudaq.set_target(target, option="mqpu,fp32")
        effective_qpu_count = (
            cudaq.get_target().num_qpus() if qpu_count == -1 else qpu_count
        )
        logger.info(
            f"Quantum target set to: {target} with QPU count: {effective_qpu_count}"
        )
        return effective_qpu_count

    qpu_count = _configure_quantum_target(device, qpu_count)

    if device not in {"cpu", "gpu"}:
        raise ValueError("Device must be either 'cpu' or 'gpu'.")
    torch_device: torch.device = torch.device(
        "cuda:0" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    if choose_best_val_epoch:
        # Choose the best validation epoch
        best_val_loss_index = (
            checkpoint["val_losses"].index(min(checkpoint["val_losses"])) + 1
        )
        best_model_filename = f"model_epoch_{best_val_loss_index}.pt"
        selected_transformer_params = os.path.join(
            os.path.dirname(checkpoint_path), best_model_filename
        )
        logger.info(
            f"Sampling on {best_model_filename} with val_loss {min(checkpoint['val_losses'])}"
        )
        checkpoint = torch.load(
            selected_transformer_params, map_location="cpu", weights_only=False
        )
    else:
        logger.info(
            f"Sampling on {os.path.basename(checkpoint_path)} with val_loss {checkpoint['val_losses'][-1]}"
        )

    property_names = [
        "MW",
        "HBA",
        "HBD",
        "nRot",
        "nRing",
        "nHet",
        "TPSA",
        "LogP",
        "StereoCenters",
    ]
    training_data_properties = pd.read_csv(
        training_data_path, usecols=property_names
    ).to_numpy()

    # No conditions are specified in the first experiment. This experiment is to see how well the model can learn a training distribution
    conditions = [[np.nan] * len(property_names)]
    save_filenames = [os.path.join(save_path, "no_conditions.csv")]

    # The center and spread of the training data properties are calculated
    if central_tendency == "mean":
        centers = np.mean(training_data_properties, axis=0)
    elif central_tendency == "median":
        centers = np.median(training_data_properties, axis=0)
    else:
        raise ValueError("central_tendency must be 'mean' or 'median'")

    if variation == "std":
        spreads = np.std(training_data_properties, axis=0)
    elif variation == "iqr":
        q25 = np.percentile(training_data_properties, 25, axis=0)
        q75 = np.percentile(training_data_properties, 75, axis=0)
        spreads = q75 - q25
    else:
        raise ValueError("variation must be 'std' or 'iqr'")

    # Generate upper and lower condition targets for each property
    targets = [None]  # for the no-conditions case
    for i, name in enumerate(property_names):
        upper = [np.nan] * len(property_names)
        lower = [np.nan] * len(property_names)

        upper_target = centers[i] + property_variation_factor * spreads[i]
        lower_candidate = centers[i] - property_variation_factor * spreads[i]

        # Disallow negative values for properties (except LogP)
        if name != "LogP":
            lower_target = max(0, lower_candidate)
        else:
            lower_target = lower_candidate

        # Create condition vectors with only one property manipulated
        upper[i] = upper_target
        lower[i] = lower_target

        # Store in our tracking list
        targets.append(upper_target)
        targets.append(lower_target)

        # Add to conditions
        conditions.append(upper)
        save_filenames.append(os.path.join(save_path, f"upper_{name}.csv"))

        conditions.append(lower)
        save_filenames.append(os.path.join(save_path, f"lower_{name}.csv"))

    validity = []
    uniqueness = []
    novelty = []
    sampled_averages = [None]  # for no-condition experiment

    for i, conds in enumerate(conditions):
        logger.info(f"Running {save_filenames[i]}")
        v, u, n = generate_smiles(
            checkpoint_path=checkpoint_path,
            save_dir=save_filenames[i],
            num_of_model_queries=num_samples,
            sampling_batch_size=sampling_batch_size,
            MW=conds[0],
            HBA=conds[1],
            HBD=conds[2],
            nRot=conds[3],
            nRing=conds[4],
            nHet=conds[5],
            TPSA=conds[6],
            LogP=conds[7],
            StereoCenters=conds[8],
            imputation_method=imputation_method,
            imputation_dataset_path=training_data_path,
            dataset_novelty_check_path=training_data_path,
            device=torch_device,
            qpu_count=qpu_count,
        )

        validity.append(v)
        uniqueness.append(u)
        novelty.append(n)

        # After generation, compute the average of the relevant property in the result
        df = pd.read_csv(save_filenames[i])
        if i == 0:
            # The no_conditions experiment
            no_conditions_mean = df.mean(numeric_only=True).tolist()
        else:
            # For the i-th condition, figure out which property it corresponds to
            # property index: (i - 1) // 2 because we appended in pairs (upper, lower)
            prop_index = (i - 1) // 2
            prop_name = property_names[prop_index]
            sampled_averages.append(df[prop_name].mean())

    # Save combined CSV summary
    output_file = os.path.join(save_path, "OVERALL_SAMPLING_RESULTS.csv")

    # Pad the property_names and no_conditions_mean lists if needed
    # This is just so that we can put in the no_conditions_mean data in the CSV off
    # to the right so it is easier to read when opening it manually
    diff_len = len(save_filenames) - len(property_names)
    property_names += [""] * diff_len
    diff_len_mean = len(save_filenames) - len(no_conditions_mean)
    no_conditions_mean += [""] * diff_len_mean

    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(
            [
                "save_filenames",
                "validity",
                "uniqueness",
                "novelty",
                "sampled_averages",
                "targets",
                "",
                "property_names",
                "no_conditions_mean",
            ]
        )

        # Rows
        for i in range(len(save_filenames)):
            writer.writerow(
                [
                    save_filenames[i],
                    validity[i],
                    uniqueness[i],
                    novelty[i],
                    sampled_averages[i],
                    targets[i],
                    "",
                    property_names[i],
                    no_conditions_mean[i],
                ]
            )

    logger.info(f"Data saved to {output_file}")
    logger.info(
        f"Central tendency: {central_tendency}, Variation: {variation}, Factor: {property_variation_factor}"
    )
