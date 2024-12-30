import os
import sys
import scanpy as sc
from anndata import AnnData
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

SC_FILE_PATH = "input/scaden/pbmc_3k_filtered.h5"
ACT_FILE_PATH = "input/scaden/ACT_annotations_3k.tsv"
NUM_SAMPLES = 1000  # Number of synthetic bulk samples to generate
NUM_CELLS = 500  # Number of cells to sum for each bulk sample
EPOCHS = 50  # Number of epochs for training
BATCH_SIZE = 32  # Batch size for training
TEST_SIZE = 0.2  # For train-test split
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScadenModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ScadenModel, self).__init__()
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


def load_SC_data(file_path) -> AnnData:
    """
    Load single-cell gene expression data from H5 file.
    Returns AnnData object containing single-cell gene expression matrix.
    """
    S = sc.read_10x_h5(file_path)
    return S


def add_ACT_annotations(S: AnnData, ACT_file_path) -> AnnData:
    """
    Add ACT-based cell-type annotations to single-cell clusters.

    Steps:
    1. Load ACT file containing cluster-to-cell-type mappings.
    2. If no clustering exists, run Leiden clustering on the dataset.
    3. Map ACT-predicted cell types to clusters and add 'cell_type' to S.obs.
    """
    ACT_df = pd.read_csv(ACT_file_path, sep="\t")
    cluster_to_ct = dict(zip(ACT_df["Cluster"], ACT_df["Cell.Type"]))

    if "leiden" not in S.obs:
        sc.pp.normalize_total(S, target_sum=1e4)
        sc.pp.log1p(S)
        sc.pp.neighbors(S, n_neighbors=10, n_pcs=40)
        sc.tl.leiden(S, resolution=0.5)

    S.obs["cell_type"] = S.obs["leiden"].replace(cluster_to_ct)
    return S


def build_dataset(S: AnnData, n_cells, n_samples):
    """
    Generate synthetic pseudo-bulk samples and corresponding cell-type fractions.

    Steps:
    1. Randomly select `n_cells` cells from the single-cell data.
    2. Sum gene counts for these cells to create simulated pseudo-bulk samples.
    3. Calculate cell-type abundance fractions (C) for each simulated sample.
    """
    B = []
    C = []
    for _ in range(n_samples):
        rand_cells = np.random.choice(S.n_obs, n_cells, replace=False)
        bulk_sample = S[rand_cells, :].X.sum(axis=0).A1
        B.append(bulk_sample)

        all_celltypes = S.obs["cell_type"].unique()
        S_subsample = S.obs.iloc[rand_cells]["cell_type"]

        ct_fractions = S_subsample.value_counts(normalize=True)
        all_ct_fractions = {ct: ct_fractions.get(ct, 0.0) for ct in all_celltypes}
        C.append(list(all_ct_fractions.values()))

    B = np.log1p(np.array(B))
    C = np.array(C)
    return B, C


def train_model(model: ScadenModel, train_set: TensorDataset, test_set: TensorDataset):
    model.to(DEVICE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for e in range(EPOCHS):
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
        print(f"Epoch {e+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")


def save_model(model: ScadenModel, X_test, Y_test):
    os.makedirs("output", exist_ok=True)
    dtnum = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    model_dir = os.path.join("output", "scaden", dtnum)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save predictions and true fractions
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(DEVICE)).cpu().numpy()
    preds_file = os.path.join(model_dir, "pred_fractions.csv")
    true_fractions_file = os.path.join(model_dir, "true_fractions.csv")
    np.savetxt(preds_file, predictions, delimiter=",")
    np.savetxt(true_fractions_file, Y_test.numpy(), delimiter=",")
    print(f"Saved predictions to {preds_file}")
    print(f"Saved true fractions to {true_fractions_file}")


def eval_model(model: ScadenModel, X_test, Y_test):
    print("\nEvaluating model on Y_test:")
    X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(
        Y_test, dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test.to(DEVICE)).cpu()

    mae = torch.mean(torch.abs(Y_pred - Y_test)).item()
    rmse = torch.sqrt(torch.mean((Y_pred - Y_test) ** 2)).item()
    cosine = torch.nn.functional.cosine_similarity(Y_pred, Y_test, dim=1).mean().item()

    print(f" - MAE: {mae:.4f}")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - Cosine similarity: {cosine:.4f}")


def main(saved_model_path=None):
    S = add_ACT_annotations(load_SC_data(SC_FILE_PATH), ACT_FILE_PATH)
    print("\nLoaded single-cell data and added ACT annotations.")
    print(f" - S dims (cells x genes): {S.shape}")

    B, C = build_dataset(S, NUM_CELLS, NUM_SAMPLES)
    print("\nGenerated synthetic dataset")
    print(f" - B dims (samples x genes): {B.shape}")
    print(f" - C dims (samples x CTs): {C.shape}")

    X_train, X_test, Y_train, Y_test = train_test_split(
        B, C, test_size=TEST_SIZE, random_state=42
    )
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
    model = ScadenModel(input_dim, output_dim)
    if saved_model_path:
        model.load_state_dict(torch.load(saved_model_path))
        print(f"\nLoaded model from {saved_model_path}")

    if not saved_model_path:
        print("\nTraining model...")
        train_model(model, train_dataset, test_dataset)
        print("Training complete!")
        save_model(model, X_test, Y_test)

    eval_model(model, X_test, Y_test)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        saved_model_path = sys.argv[1]
        main(saved_model_path)
    else:
        main()
