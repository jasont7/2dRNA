import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

SC_FILE_PATH = "data/pbmc_3k_filtered.h5"
ACT_FILE_PATH = "data/ACT_annotations_3k.tsv"
NUM_SAMPLES = 1000  # Number of synthetic bulk samples to generate
NUM_CELLS = 500  # Number of cells to sum for each bulk sample
EPOCHS = 50  # Number of epochs for training
BATCH_SIZE = 32  # Batch size for training
TEST_SIZE = 0.2  # For train-test split


def load_single_cell_data(file_path) -> AnnData:
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


def build_train_set(S: AnnData, n_cells, n_samples):
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
        # Randomly select cells and sum their gene counts to create pseudo-bulk samples
        rand_cells = np.random.choice(S.n_obs, n_cells, replace=False)
        bulk_sample = S[rand_cells, :].X.sum(axis=0).A1
        B.append(bulk_sample)

        # Calculate cell-type abundance fractions for the random cells selected
        all_celltypes = S.obs["cell_type"].unique()
        S_subsample = S.obs.iloc[rand_cells]["cell_type"]

        # Dict of cell type counts normalized to sum to 1
        celltype_fractions = S_subsample.value_counts(normalize=True)

        # Impute missing cell types with 0 fraction
        all_celltype_fractions = {
            ct: celltype_fractions.get(ct, 0.0) for ct in all_celltypes
        }
        C.append(list(all_celltype_fractions.values()))

    B = np.log1p(np.array(B))  # Log-normalize bulk samples
    C = np.array(C)
    return B, C


def build_model(input_dim, output_dim):
    """
    Build Scaden-like neural network.

    1. Define a 4-layer feedforward network with batch normalization and dropout.
    2. Use linear activation in the output layer to predict cell-type abundance fractions.
    3. Compile the model using Adam optimizer and MSE loss.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(1000, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(500, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(output_dim, activation="linear"),  # No softmax
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model


def save_model_and_preds(model, X_test, y_test, history):
    """
    Save the trained model, training history, and predictions on the test set.
    """
    os.makedirs("output", exist_ok=True)

    model_path = os.path.join("output", "scaden_model.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    history_path = os.path.join("output", "training_history.npy")
    np.save(history_path, history.history)
    print(f"Saved training history to {history_path}")

    y_pred = model.predict(X_test)
    predictions_file = os.path.join("output", "predicted_fractions.csv")
    true_fractions_file = os.path.join("output", "true_fractions.csv")
    np.savetxt(predictions_file, y_pred, delimiter=",")
    np.savetxt(true_fractions_file, y_test, delimiter=",")
    print(f"Saved predicted fractions to {predictions_file}")
    print(f"Saved true fractions to {true_fractions_file}")


def eval_model(model, X_test, y_test):
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print("Evaluation results:\n", f"Test Loss: {test_loss}, Test MAE: {test_mae}\n")


def main():
    S = load_single_cell_data(SC_FILE_PATH)
    print("\nLoaded single-cell data")
    print(f"Dataset shape (cells x genes): {S.shape}", "\n")

    S = add_ACT_annotations(S, ACT_FILE_PATH)
    print("Added cell-type annotations\nSample:")
    print(S.obs.head(), "\n")

    B, C = build_train_set(S, NUM_CELLS, NUM_SAMPLES)
    print(
        "Generated synthetic dataset\n",
        f"  Bulk matrix shape (samples x genes): {B.shape}\n",
        f"  CT abundance matrix shape (samples x CTs): {C.shape}\n",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        B, C, test_size=TEST_SIZE, random_state=42
    )

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = build_model(input_dim, output_dim)
    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=TEST_SIZE,
        verbose=2,
    )
    print("Model training complete!\n")
    save_model_and_preds(model, X_test, y_test, history)

    eval_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
