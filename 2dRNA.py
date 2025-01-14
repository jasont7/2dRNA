import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 100)
np.set_printoptions(linewidth=120)
np.set_printoptions(precision=4, suppress=True)

BULK_PATH = "input/2dRNA/group1/bulk_RawCounts.tsv"
SC_DIR_PATH = "input/2dRNA/group1/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    bulk_df = pd.read_csv(BULK_PATH, sep="\t")

    print("B Matrix (Tissue GEPs) Sample:\n")
    print(bulk_df.iloc[:, :6].head(5))
    print("\n----------------------------------------------")
    print(
        f"\nB DIMENSIONS: rows (genes) = {bulk_df.shape[0]}, columns (patients) = {bulk_df.shape[1]}"
    )

    sc_path = SC_DIR_PATH + "scRNA_CT1_top200_RawCounts.tsv"
    sc_df = pd.read_csv(sc_path, sep="\t")

    print("S Matrix (Cell GEPs) Sample:\n")
    print(sc_df.iloc[:, :12].head(5))
    print("\n----------------------------------------------")
    print(
        f"\nS DIMENSIONS: rows (patients x cells) = {sc_df.shape[0]}, columns (genes) = {sc_df.shape[1]}"
    )

    sc_metadata_path = SC_DIR_PATH + "scRNA_CT1_top200_Metadata.tsv"
    sc_metadata_df = pd.read_csv(sc_metadata_path, sep="\t")

    print("S Metadata Matrix Sample:\n")
    print(sc_metadata_df.head(5))
    print("\n----------------------------------------------")
    print("S Metadata Info:\n")
    sc_metadata_df.info()
    print("----------------------------------------------")
    print(
        f"\nS METADATA DIMENSIONS: rows (patients x cells) = {sc_metadata_df.shape[0]}, columns (metadata) = {sc_metadata_df.shape[1]}\n"
    )

    return bulk_df, sc_df, sc_metadata_df


def process_bulk(bulk: pd.DataFrame, sc: pd.DataFrame, sc_metadata: pd.DataFrame):
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
    print(f"B dims (patients x genes): {B.shape}")

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

    C_augs = np.array(C_augs)
    # Flatten to 2D so each row is an augmentation for a specific patient
    C_flat = C_augs.reshape(-1, C_augs.shape[2])
    print(f"C dims ((patients * n_augs) x CTs): {C_flat.shape}")
    print(f"Processed patients: {len(processed_patients)}")

    return C_flat, processed_patients


def data_prep_pipeline(n_aug=30, aug_ratio=0.9):
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
    print("C sample:\n", C_flat[10:20, :])

    B_filtered = B[np.isin(patient_ids, processed_patient_ids)]
    print(f"Filtered B dims (patients x genes): {B_filtered.shape}")

    B_aug = np.repeat(B_filtered, n_aug, axis=0)  # Repeat B for each augmentation
    print(f"Flattened B dims ((patients * n_augs) x genes): {B_aug.shape}")

    return train_test_split(B_aug, C_flat, test_size=0.2, random_state=42)


class Model2dRNA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model2dRNA, self).__init__()
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

    def train(self, train_set, test_set, epochs, batch_size):
        self.to(DEVICE)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for e in range(epochs):
            self.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, y_val in test_loader:
                    X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                    val_outputs = self(X_val)
                    val_loss += criterion(val_outputs, y_val).item()
            print(
                f"Epoch {e+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    def save(self, X_test, Y_test):
        os.makedirs("output", exist_ok=True)
        dtnum = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
        model_dir = os.path.join("output", "2dRNA", dtnum)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.pth")
        torch.save(self.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        # Save predictions and true fractions
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            predictions = self(X_test.to(DEVICE)).cpu().numpy()
        preds_file = os.path.join(model_dir, "pred_fractions.csv")
        true_fractions_file = os.path.join(model_dir, "true_fractions.csv")
        np.savetxt(preds_file, predictions, delimiter=",")
        np.savetxt(true_fractions_file, Y_test.numpy(), delimiter=",")
        print(f"Saved predictions to {preds_file}")
        print(f"Saved true fractions to {true_fractions_file}")

    def eval(self, X_test, Y_test):
        print("\nEvaluating model on Y_test:")
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            Y_pred = self(X_test.to(DEVICE)).cpu()

        target_min = Y_test.min()
        target_max = Y_test.max()
        target_mean = Y_test.mean()

        mae = torch.mean(torch.abs(Y_pred - Y_test)).item()
        rmse = torch.sqrt(torch.mean((Y_pred - Y_test) ** 2)).item()
        cosine = (
            torch.nn.functional.cosine_similarity(Y_pred, Y_test, dim=1).mean().item()
        )

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


def main():
    X_train, X_test, Y_train, Y_test = data_prep_pipeline()

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
    model = Model2dRNA(input_dim, output_dim)
    epochs = 150
    batch_size = 32

    saved_model_path = None  # "output/2dRNA/20241229_1515/model.pth"
    if saved_model_path and os.path.exists(saved_model_path):
        model.load_state_dict(torch.load(saved_model_path))
        print(f"Loaded model from {saved_model_path}")
    else:
        print("Training model...")
        model.train(model, train_dataset, test_dataset, epochs, batch_size)
        print("Training complete!")
        model.save(model, X_test, Y_test)

    model.eval(model, X_test, Y_test)
