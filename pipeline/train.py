import sys
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import warnings
from process import data_prep_pipeline
from predict import predict_ensemble_avg
from SimpleDNN import (
    SimpleDNN,
    train_simple_dnn,
    hyperparam_search_simple_dnn,
)
from utils import save_to_disk, eval_metrics

warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=120)
np.set_printoptions(precision=4, suppress=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BULK_PATH = "input/2dRNA/group1/bulk_RawCounts.tsv"
SC_PATH = "input/2dRNA/group1/scRNA_CT2_top500_RawCounts.tsv"
SC_METADATA_PATH = "input/2dRNA/group1/scRNA_CT2_top500_Metadata.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pipeline_simple_dnn():
    # B is model input, C is model output
    B, C = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )
    X_train, X_test, Y_train, Y_test = train_test_split(B, C, test_size=0.2)

    # Hyperparam search for finding best SimpleDNN architecture
    model_param_grid = {
        "hidden_layers": [
            ([256, 64], [0.1, 0.1]),
            ([256, 64], [0.05, 0.05]),
            ([256, 64], [0.1, 0]),
            ([256, 128], [0.1, 0.1]),
            ([256, 128], [0.05, 0.05]),
            ([256, 128], [0.1, 0]),
            ([256, 256], [0.1, 0.1]),
            ([256, 256], [0.05, 0.05]),
            ([256, 256], [0.1, 0]),
            ([512, 128], [0.1, 0.1]),
            ([512, 128], [0.05, 0.05]),
            ([512, 128], [0.1, 0]),
            ([512, 256], [0.1, 0.1]),
            ([512, 256], [0.05, 0.05]),
            ([512, 256], [0.1, 0]),
            ([512, 512], [0.1, 0.1]),
            ([512, 512], [0.05, 0.05]),
            ([512, 512], [0.1, 0]),
            ([1024, 256], [0.1, 0.1]),
            ([1024, 256], [0.05, 0.05]),
            ([1024, 256], [0.1, 0]),
            ([1024, 512], [0.1, 0.1]),
            ([1024, 512], [0.05, 0.05]),
            ([1024, 512], [0.1, 0]),
            ([1024, 1024], [0.1, 0.1]),
            ([1024, 1024], [0.05, 0.05]),
            ([1024, 1024], [0.1, 0]),
        ]
    }
    best_model_params = hyperparam_search_simple_dnn(B, C, model_param_grid, k=3)

    final_model = SimpleDNN(
        input_dim=B.shape[1],
        output_dim=C.shape[1],
        hidden_layers=best_model_params["hidden_layers"],
    ).to(DEVICE)

    print("\nTraining final model with best parameters...")
    train_simple_dnn(final_model, X_train, X_test, Y_train, Y_test)
    save_to_disk(final_model, X_test, Y_test, "SimpleDNN")

    print(eval_metrics(Y_test, model=final_model, X_test=X_test))


def pipeline_scaden():
    # B is model input, C is model output
    B, C = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )
    X_train, X_test, Y_train, Y_test = train_test_split(B, C, test_size=0.2)

    architectures = {
        "m256": ([256, 128, 64, 32], [0, 0, 0, 0]),
        "m512": ([512, 256, 128, 64], [0, 0.3, 0.2, 0.1]),
        "m1024": ([1024, 512, 256, 128], [0, 0.6, 0.3, 0.1]),
    }
    ensemble_models = []
    for name, hidden_layers in architectures.items():
        model = SimpleDNN(
            input_dim=B.shape[1],
            output_dim=C.shape[1],
            hidden_layers=hidden_layers,
        ).to(DEVICE)
        ensemble_models.append((name, model))

    for name, model in ensemble_models:
        print(f"\nTraining {name} model...")
        train_simple_dnn(
            model,
            X_train,
            X_test,
            Y_train,
            Y_test,
            batch_size=128,
            criterion=nn.L1Loss(),
            lr=0.0001,
            epochs=5000,
            patience=500,
        )
        save_to_disk(model, X_test, Y_test, name)

    ensemble_predictions = predict_ensemble_avg(ensemble_models, X_test)
    print(eval_metrics(Y_test, Y_pred=ensemble_predictions))


if __name__ == "__main__":
    if sys.argv[1] == "SimpleDNN":
        pipeline_simple_dnn()
    elif sys.argv[1] == "Scaden":
        pipeline_scaden()
    else:
        raise ValueError("Invalid pipeline argument.")
