import os
import sys
import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.spatial import distance
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
        # Randomly select cells and sum their gene counts to create pseudo-bulk samples
        rand_cells = np.random.choice(S.n_obs, n_cells, replace=False)
        bulk_sample = S[rand_cells, :].X.sum(axis=0).A1
        B.append(bulk_sample)

        # Calculate cell-type abundance fractions for the random cells selected
        all_celltypes = S.obs["cell_type"].unique()
        S_subsample = S.obs.iloc[rand_cells]["cell_type"]

        # Dict of cell type counts normalized to sum to 1
        ct_fractions = S_subsample.value_counts(normalize=True)

        # Impute missing cell types with 0 fraction
        all_ct_fractions = {ct: ct_fractions.get(ct, 0.0) for ct in all_celltypes}
        C.append(list(all_ct_fractions.values()))

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


def save_training(model, X_test, Y_test, history):
    """
    Save the trained model, training history, and predictions on the test set.
    """
    os.makedirs("output", exist_ok=True)

    dtnum = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    model_dir = os.path.join("output", "scaden", dtnum)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"model.keras")
    model.save(model_path)
    print(f"Saved model to {model_path}")

    history_path = os.path.join(model_dir, f"history.npy")
    np.save(history_path, history.history)
    print(f"Saved training history to {history_path}")

    y_pred = model.predict(X_test)
    preds_file = os.path.join(model_dir, f"pred_fractions.csv")
    true_fractions_file = os.path.join(model_dir, f"true_fractions.csv")
    np.savetxt(preds_file, y_pred, delimiter=",")
    np.savetxt(true_fractions_file, Y_test, delimiter=",")
    print(f"Saved predicted fractions to {preds_file}")
    print(f"Saved true fractions to {true_fractions_file}")


def load_model(model_path):
    """
    Load a pre-trained model from a Keras model file.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model successfully loaded from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please check the path.")
    except tf.errors.OpError as op_err:
        print(f"TensorFlow operation error while loading model: {op_err}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
    sys.exit(1)


def eval_model(model, X_test, Y_test):
    print("\nEvaluating model on Y_test:")

    mae = tf.keras.metrics.MeanAbsoluteError()
    mae.update_state(Y_test, model.predict(X_test, verbose=0))
    mae_score = mae.result().numpy()
    print(f" - MAE score: {mae_score}")

    rmse = tf.keras.metrics.RootMeanSquaredError()
    rmse.update_state(Y_test, model.predict(X_test, verbose=0))
    rmse_score = rmse.result().numpy()
    print(f" - RMSE score: {rmse_score}")

    cosine = tf.keras.metrics.CosineSimilarity(axis=1)
    cosine.update_state(Y_test, model.predict(X_test, verbose=0))
    cosine_score = cosine.result().numpy()
    print(f" - Cosine similarity: {cosine_score}")

    # r2 = tf.keras.metrics.RSquared()
    # r2.update_state(Y_test, model.predict(X_test))
    # r2_score = r2.result().numpy()
    # print(f" - R^2 score: {r2_score}")

    # jsd = distance.jensenshannon(Y_test, model.predict(X_test, verbose=0)) ** 2
    # print(f" - Jensen-Shannon divergence: {jsd}")


def main(saved_model=None):
    S = load_SC_data(SC_FILE_PATH)
    print("\nLoaded single-cell data")
    print(f" - S matrix shape (cells x genes): {S.shape}")
    S = add_ACT_annotations(S, ACT_FILE_PATH)
    print(" - Added cell-type annotations to S")

    B, C = build_dataset(S, NUM_CELLS, NUM_SAMPLES)
    print("\nGenerated synthetic dataset")
    print(f" - B matrix shape (samples x genes): {B.shape}")
    print(f" - C matrix shape (samples x CTs): {C.shape}")

    X_train, X_test, Y_train, Y_test = train_test_split(
        B, C, test_size=TEST_SIZE, random_state=42
    )

    if saved_model:
        model = saved_model
    else:
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]
        model = build_model(input_dim, output_dim)
        print("\nTraining model...\n")
        history = model.fit(
            X_train,
            Y_train,
            validation_data=(X_test, Y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=TEST_SIZE,
            verbose=2,
        )
        print("\nModel training complete!")
        save_training(model, X_test, Y_test, history)

    eval_model(model, X_test, Y_test)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        saved_model_path = sys.argv[1]
        model = load_model(saved_model_path)
        print("\nLoaded saved model")
        main(model)
    else:
        print("\nTraining new model")
        main()
