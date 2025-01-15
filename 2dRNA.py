import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import warnings

warnings.filterwarnings("ignore")
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 1000)
# pd.set_option("display.max_colwidth", 100)
np.set_printoptions(linewidth=120)
np.set_printoptions(precision=4, suppress=True)


########## CONSTANTS (please update) ##########
BULK_PATH = "input/2dRNA/group1/bulk_RawCounts.tsv"
SC_PATH = "input/2dRNA/group1/scRNA_CT1_top200_RawCounts.tsv"
SC_METADATA_PATH = "input/2dRNA/group1/scRNA_CT1_top200_Metadata.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_AUG = 30
AUG_RATIO = 0.9
###############################################


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)


class ScadenNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ScadenNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(0.3),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(0.3),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, output_dim),
        )

    def forward(self, x):
        return self.model(x)


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


def process_SC(sc_metadata: pd.DataFrame, patient_ids: np.ndarray, n_aug, aug_ratio):
    """
    Derive augmented C matrices with random sampling (not stratified).

    Args:
        sc_metadata (pd.DataFrame): Single-cell metadata containing patient and cell type information.
        patient_ids (np.ndarray): Array of patient IDs in bulk data.
        n_aug (int): Number of augmentations per patient.
        aug_ratio (float): Fraction of cells to sample for each augmentation (e.g., 0.9 for 90%).

    Returns:
        np.ndarray: Augmented C matrices (patients x n_augs x cell types).
        list: Successfully processed patient IDs.
    """
    ct_labels = sc_metadata["cell_type_1"].dropna().unique()
    C_augs = []  # Augmentations for all patients
    processed_patients = []  # Successfully processed patient IDs

    for pid in patient_ids:
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
    C_flat, processed_patient_ids = process_SC(
        sc_metadata_df, patient_ids, n_aug, aug_ratio
    )

    # Keep only patients with both B and C data
    B_filtered = B[np.isin(patient_ids, processed_patient_ids)]
    print(f"\nFiltered B dims (patients x genes): {B_filtered.shape}")

    # Repeat B_i for each augmentation of S_i
    B_aug = np.repeat(B_filtered, n_aug, axis=0)
    print(f"Flattened B dims ((patients * n_augs) x genes): {B_aug.shape}")

    return train_test_split(B_aug, C_flat, test_size=0.2, random_state=42)


def train_model(model, train_set, test_set, epochs, batch_size, optimizer, criterion):
    """
    Train any PyTorch model on the given datasets.

    Args:
        model (nn.Module): PyTorch model to train.
        train_set (TensorDataset): Training dataset.
        test_set (TensorDataset): Validation dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function for training.

    Returns:
        None
    """

    model.to(DEVICE)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for e in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, y_val).item()
        print(f"Epoch {e+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")


def save_model(model, X_test, Y_test, model_name):
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


def evaluate_model(model, X_test, Y_test):
    print("\nEvaluating model on Y_test:")
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test.to(DEVICE)).cpu()

    target_min = Y_test.min()
    target_max = Y_test.max()
    target_mean = Y_test.mean()

    mae = torch.mean(torch.abs(Y_pred - Y_test)).item()
    rmse = torch.sqrt(torch.mean((Y_pred - Y_test) ** 2)).item()
    cosine = torch.nn.functional.cosine_similarity(Y_pred, Y_test, dim=1).mean().item()

    mae_pct_range = (mae / (target_max - target_min)) * 100
    mae_pct_mean = (mae / target_mean) * 100
    rmse_pct_range = (rmse / (target_max - target_min)) * 100
    rmse_pct_mean = (rmse / target_mean) * 100

    print(f" - Target value range: [{target_min:.4f}, {target_max:.4f}]")
    print(f" - Target value average: {target_mean:.4f}")
    print(f" - MAE: {mae:.4f}")
    print(f" - MAE as percentage of range: {mae_pct_range:.2f}%")
    print(f" - MAE as percentage of average: {mae_pct_mean:.2f}%")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - RMSE as percentage of range: {rmse_pct_range:.2f}%")
    print(f" - RMSE as percentage of average: {rmse_pct_mean:.2f}%")
    print(f" - Cosine similarity: {cosine:.4f}")


def linear_pipeline():
    X_train, X_test, Y_train, Y_test = data_prep_pipeline(N_AUG, AUG_RATIO)

    # Normalize data
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

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    model = LinearModel(input_dim, output_dim)

    epochs = 300
    batch_size = 32
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)  # Use robust loss

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    model.apply(init_weights)

    print("Training linear model...")
    train_model(
        model,
        train_dataset,
        test_dataset,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        criterion=criterion,
    )
    print("Linear model training complete!")
    save_model(model, X_test, Y_test, "linear")
    evaluate_model(model, X_test, Y_test)


def scaden_pipeline():
    X_train, X_test, Y_train, Y_test = data_prep_pipeline(N_AUG, AUG_RATIO)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    model = ScadenNN(input_dim, output_dim)

    saved_model_path = None  # "output/2dRNA/20241229_1515/model.pth"
    if saved_model_path and os.path.exists(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path))
        print(f"Loaded model from {saved_model_path}")
    else:
        epochs = 150
        batch_size = 32
        optimizer = (
            optim.Adam(model.parameters(), lr=0.001)
            if isinstance(model, ScadenNN)
            else optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
        )
        criterion = nn.MSELoss()

        print("\nTraining Scaden model...")
        train_model(
            model,
            train_dataset,
            test_dataset,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
        )
        print("Training complete!")
        save_model(model, X_test, Y_test, "scaden")

    evaluate_model(model, X_test, Y_test)


if __name__ == "__main__":
    if sys.argv[1] == "linear":
        linear_pipeline()
    elif sys.argv[1] == "nonlinear":
        scaden_pipeline()
    else:
        raise ValueError("Invalid pipeline argument. Choose 'linear' or 'nonlinear'.")
