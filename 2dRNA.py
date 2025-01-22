import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import wasserstein_distance
import datetime
import warnings
import heapq

warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=120)
np.set_printoptions(precision=4, suppress=True)


### CONSTANTS ###
BULK_PATH = "input/2dRNA/group1/bulk_RawCounts.tsv"
SC_PATH = "input/2dRNA/group1/scRNA_CT1_top200_RawCounts.tsv"
SC_METADATA_PATH = "input/2dRNA/group1/scRNA_CT1_top200_Metadata.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### DATA PREPROCESSING ###


def load_data():
    bulk_df = pd.read_csv(BULK_PATH, sep="\t")
    sc_df = pd.read_csv(SC_PATH, sep="\t")
    sc_metadata_df = pd.read_csv(SC_METADATA_PATH, sep="\t")

    print("B Matrix (Tissue GEPs) Sample:\n")
    print(bulk_df.iloc[:, :6].head(5))
    print(
        f"\nB DIMENSIONS: rows (genes) = {bulk_df.shape[0]}, columns (patients) = {bulk_df.shape[1]}"
    )
    print("\n----------------------------------------------\n")

    print("S Matrix (Cell GEPs) Sample:\n")
    print(sc_df.iloc[:, :12].head(5))
    print(
        f"\nS DIMENSIONS: rows (patients x cells) = {sc_df.shape[0]}, columns (genes) = {sc_df.shape[1]}"
    )
    print("\n----------------------------------------------\n")

    print("S Metadata Matrix Sample:\n")
    print(sc_metadata_df.head(5))
    print(
        f"\nS METADATA DIMENSIONS: rows (patients x cells) = {sc_metadata_df.shape[0]}, columns (metadata) = {sc_metadata_df.shape[1]}"
    )
    print("\n----------------------------------------------\n")

    return bulk_df, sc_df, sc_metadata_df


def process_bulk(bulk: pd.DataFrame, sc: pd.DataFrame, sc_metadata: pd.DataFrame):
    """
    Process bulk data to keep only common genes with single-cell data, and
    assert that all bulk samples have a corresponding single-cell sample.

    Args:
        bulk (pd.DataFrame): Bulk GEP data (genes x patients).
        sc (pd.DataFrame): Single-cell GEP data ((patients * cells) x genes).
        sc_metadata (pd.DataFrame): Single-cell metadata ((patients * cells) x metadata).

    Returns:
        tuple: Processed B matrix (patients x genes) and patient IDs.
    """

    # Filter B to keep only common genes with S
    bulk_genes_all = bulk["gene_symbol"].str.strip().str.lower()
    common_genes = set(bulk_genes_all).intersection(
        sc.columns[2:].str.strip().str.lower()
    )
    filtered_bulk = bulk[bulk_genes_all.isin(common_genes)].drop_duplicates(
        subset="gene_symbol", keep="first"
    )
    filtered_bulk_vals = filtered_bulk.iloc[:, 2:]  # drop gene_id and gene_symbol cols

    # Normalize and convert to np array
    B = np.log1p(filtered_bulk_vals.values.T)
    print(f"Filtered B dims (patients x genes): {B.shape}\n")

    # Assert that patient IDs in S match B
    sc_patient_ids = sc_metadata["patient_id"].unique()
    bulk_patient_ids = filtered_bulk_vals.columns
    if not all(i in bulk_patient_ids for i in sc_patient_ids):
        raise ValueError("Patient IDs in S do not match B. Check mapping.")

    return B, bulk_patient_ids


def process_single_cell(sc_metadata: pd.DataFrame, pids: np.ndarray, n_aug, aug_ratio):
    """
    Derive augmented C matrices with random sampling (not stratified).

    Args:
        sc_metadata (pd.DataFrame): Single-cell metadata containing patient and cell type information.
        pids (np.ndarray): Array of patient IDs in bulk data.
        n_aug (int): Number of augmentations per patient.
        aug_ratio (float): Fraction of cells to sample for each augmentation (e.g., 0.9 for 90%).

    Returns:
        np.ndarray: Augmented C matrices (patients x n_augs x cell types).
        list: Successfully processed patient IDs.
    """
    ct_labels = sc_metadata["cell_type_1"].dropna().unique()
    C_augs = []  # Augmentations for all patients
    processed_patients = []  # Successfully processed patient IDs

    for pid in pids:
        print(f"Augmenting Patient {pid}")
        patient_cells = sc_metadata[sc_metadata["patient_id"] == pid]
        if patient_cells.empty:
            print(f"  Skipping (no cells)")
            continue

        patient_augs = []

        for _ in range(n_aug):
            # Randomly sample a subset of all cells
            n_sample = max(1, int(len(patient_cells) * aug_ratio))
            sampled_cells = patient_cells.sample(
                n=n_sample, replace=False, random_state=None
            )

            # Calculate cell type fractions for this augmentation
            ct_fractions = sampled_cells["cell_type_1"].value_counts(normalize=True)
            all_ct_fractions = {ct: ct_fractions.get(ct, 0.0) for ct in ct_labels}
            patient_augs.append(list(all_ct_fractions.values()))

        C_augs.append(patient_augs)
        processed_patients.append(pid)

    print(f"\nProcessed patients: {len(processed_patients)}")

    C_augs = np.array(C_augs)
    # Flatten to 2D so each row is an augmentation for a specific patient
    C_flat = C_augs.reshape(-1, C_augs.shape[2])
    print(f"C dims ((patients * n_augs) x CTs): {C_flat.shape}")

    return C_flat, processed_patients


def data_prep_pipeline(n_aug, aug_ratio):
    """
    Pipeline for preparing all necessary data for training.

    Args:
        n_aug (int): Number of augmentations per patient.
        aug_ratio (float): Fraction of cells to sample for each augmentation (e.g., 0.9 for 90%).

    Returns:
        tuple: Train-test split of B and C matrices (X_train, X_test, Y_train, Y_test).
    """

    bulk_df, sc_df, sc_metadata_df = load_data()
    B, patient_ids = process_bulk(bulk_df, sc_df, sc_metadata_df)
    C_flat, processed_patient_ids = process_single_cell(
        sc_metadata_df, patient_ids, n_aug, aug_ratio
    )

    # Keep only patients with both B and C data
    B_filtered = B[np.isin(patient_ids, processed_patient_ids)]
    print(f"\nFiltered B dims (patients x genes): {B_filtered.shape}")

    # Repeat B_i for each augmentation of S_i
    B_aug = np.repeat(B_filtered, n_aug, axis=0)
    print(f"Flattened B dims ((patients * n_augs) x genes): {B_aug.shape}")

    return train_test_split(B_aug, C_flat, test_size=0.2, random_state=42)


### MODELLING ###


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)


class NonlinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate):
        super(NonlinearModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for layer_size in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, layer_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = layer_size
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    for e in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, Y_val).item()
        print(
            f"Epoch {e+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
    return val_loss


def hyperparam_search_kf_cv(model_class, param_grid, X, Y, k=5):
    """Hyperparameter search using k-fold cross-validation with ranked output."""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    best_params = None
    best_combined_loss = float("inf")
    param_heap = []

    for idx, params in enumerate(ParameterGrid(param_grid)):
        print(f"\nParam Set: {idx+1}/{len(ParameterGrid(param_grid))}")
        model_params = {
            k: v for k, v in params.items() if k not in ["lr", "epochs", "criterion"]
        }

        fold_losses = []
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(Y_train, dtype=torch.float32),
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(Y_val, dtype=torch.float32),
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            model = model_class(**model_params).to(DEVICE)
            val_loss = train_model(
                model,
                train_loader,
                val_loader,
                optimizer=optim.Adam(model.parameters(), params["lr"]),
                criterion=params["criterion"],
                epochs=params["epochs"],
            )
            fold_losses.append(val_loss)

        avg_loss = sum(fold_losses) / len(fold_losses)
        print(f"Avg Loss for Params {params}: {avg_loss:.4f}")

        heapq.heappush(param_heap, (avg_loss, params))

        # Update best params
        if avg_loss < best_combined_loss:
            best_combined_loss = avg_loss
            best_params = params

    print(f"\nBest Params: {best_params} with Loss: {best_combined_loss:.4f}")

    print("\nRanked Parameter Sets:")
    ranked_params = sorted(param_heap)
    for rank, (loss, param_set) in enumerate(ranked_params, start=1):
        print(f"Rank {rank}: Loss = {loss:.4f}, Params = {model_params}")

    return best_params


### CUSTOM LOSS FUNCTIONS ###


def kl_divergence_loss(y_true, y_pred):
    epsilon = 1e-8
    y_pred = torch.clamp(y_pred, min=epsilon, max=1)
    y_true = torch.clamp(y_true, min=epsilon, max=1)
    return torch.sum(y_true * torch.log(y_true / y_pred), dim=-1).mean()


def wasserstein_loss(y_true, y_pred):
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    distances = [
        wasserstein_distance(y_true_np[i], y_pred_np[i]) for i in range(len(y_true_np))
    ]
    return torch.tensor(distances, dtype=torch.float32, device=y_true.device).mean()


def aitchison_distance_loss(y_true, y_pred):
    epsilon = 1e-8
    y_true = torch.clamp(y_true, min=epsilon, max=1)
    y_pred = torch.clamp(y_pred, min=epsilon, max=1)
    log_ratio_diff = torch.log(y_true / y_true.prod(dim=-1, keepdim=True)) - torch.log(
        y_pred / y_pred.prod(dim=-1, keepdim=True)
    )
    return torch.sqrt(torch.mean(log_ratio_diff**2, dim=-1)).mean()


def huber_loss(y_true, y_pred, delta=0.1):
    return F.huber_loss(y_pred, y_true, delta=delta).mean()


def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = 1e-8
    y_pred = torch.clamp(y_pred, min=epsilon, max=1 - epsilon)
    ce = -y_true * torch.log(y_pred)
    weight = (1 - y_pred) ** gamma
    return (weight * ce).sum(dim=-1).mean()


def save_to_disk(model, X_test, Y_test, model_name):
    """Save model, predictions, and true fractions to disk."""
    os.makedirs("output", exist_ok=True)
    dtnum = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    model_dir = os.path.join("output", "2dRNA", dtnum)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_name} to {model_path}")

    # Save predictions and true fractions
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(DEVICE)).cpu().numpy()
    preds_file = os.path.join(model_dir, f"{model_name}_pred_fractions.csv")
    true_fractions_file = os.path.join(model_dir, f"{model_name}_true_fractions.csv")
    np.savetxt(preds_file, predictions, delimiter=",")
    np.savetxt(true_fractions_file, Y_test.numpy(), delimiter=",")
    print(f"Saved predictions to {preds_file}")
    print(f"Saved true fractions to {true_fractions_file}")


def print_model_eval_details(model, X_test, Y_test):
    """Evaluate model on test set."""
    print("\nEvaluating model on Y_test:")
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test.to(DEVICE)).cpu()

    target_min = Y_test.min()
    target_max = Y_test.max()
    target_mean = Y_test.mean()
    target_median = torch.median(Y_test).item()

    mae = torch.mean(torch.abs(Y_pred - Y_test)).item()
    rmse = torch.sqrt(torch.mean((Y_pred - Y_test) ** 2)).item()
    cosine = torch.nn.functional.cosine_similarity(Y_pred, Y_test, dim=1).mean().item()

    mae_pct_range = (mae / (target_max - target_min)) * 100
    mae_pct_mean = (mae / target_mean) * 100
    mae_pct_median = (mae / target_median) * 100
    rmse_pct_range = (rmse / (target_max - target_min)) * 100
    rmse_pct_mean = (rmse / target_mean) * 100
    rmse_pct_median = (rmse / target_median) * 100

    print(f" - Target value range: [{target_min:.4f}, {target_max:.4f}]")
    print(f" - Target value average: {target_mean:.4f}")
    print(f" - MAE: {mae:.4f}")
    print(f" - MAE as percentage of range: {mae_pct_range:.2f}%")
    print(f" - MAE as percentage of average: {mae_pct_mean:.2f}%")
    print(f" - MAE as percentage of median: {mae_pct_median:.2f}%")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - RMSE as percentage of range: {rmse_pct_range:.2f}%")
    print(f" - RMSE as percentage of average: {rmse_pct_mean:.2f}%")
    print(f" - RMSE as percentage of median: {rmse_pct_median:.2f}%")
    print(f" - Cosine similarity: {cosine:.4f}")


### PIPELINES ###


def nonlinear_pipeline():
    X_train, X_test, Y_train, Y_test = data_prep_pipeline(n_aug=30, aug_ratio=0.9)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    saved_model_path = None  # "output/2dRNA/20250116_..../model.pth"
    if saved_model_path and os.path.exists(saved_model_path):
        model = NonlinearModel(input_dim=X_train.shape[1], output_dim=Y_train.shape[1])
        model.load_state_dict(torch.load(saved_model_path))
        print(f"Loaded model from {saved_model_path}")
    else:
        param_grid = {
            "input_dim": [X_train.shape[1]],
            "output_dim": [Y_train.shape[1]],
            "lr": [0.001],
            "epochs": [150],
            "criterion": [nn.MSELoss()],  # TODO: Add more loss functions here
            "dropout_rate": [0.05, 0.1, 0.2],
            "hidden_layers": [
                [32],
                [64],
                [128],
                [32, 32],
                [64, 32],
                [128, 32],
                [128, 32, 64],
            ],
        }
        best_params = hyperparam_search_kf_cv(
            NonlinearModel, param_grid, X_train, Y_train, k=3
        )
        model_params = {
            k: v
            for k, v in best_params.items()
            if k not in ["lr", "epochs", "criterion"]
        }
        model = NonlinearModel(**model_params).to(DEVICE)

        print("\nTraining final model with best parameters...")
        train_model(
            model,
            train_loader,
            test_loader,
            optimizer=optim.Adam(model.parameters(), best_params["lr"]),
            criterion=best_params["criterion"],
            epochs=150,
        )
        save_to_disk(model, X_test, Y_test, "nonlinear")

    print_model_eval_details(model, X_test, Y_test)


def linear_pipeline():
    X_train, X_test, Y_train, Y_test = data_prep_pipeline(n_aug=30, aug_ratio=0.9)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    model = LinearModel(input_dim, output_dim)

    print("Training linear model...")
    train_model(
        model,
        train_loader,
        test_loader,
        optimizer=optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4),
        criterion=nn.HuberLoss(delta=1.0),
        epochs=300,
    )
    print("Linear model training complete!")
    save_to_disk(model, X_test, Y_test, "linear")
    print_model_eval_details(model, X_test, Y_test)


if __name__ == "__main__":
    if sys.argv[1] == "linear":
        linear_pipeline()
    elif sys.argv[1] == "nonlinear":
        nonlinear_pipeline()
    else:
        raise ValueError("Invalid pipeline argument. Choose 'linear' or 'nonlinear'.")
