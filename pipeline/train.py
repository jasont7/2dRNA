import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
import warnings
from process import data_prep_pipeline
from SimpleDNN import (
    SimpleDNN,
    SimpleLinear,
    train_simple_dnn,
    hyperparam_search_simple_dnn,
)
from Scaden import ScadenModel, train_scaden_ensemble, predict_ensemble
from utils import save_to_disk, print_eval

warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=120)
np.set_printoptions(precision=4, suppress=True)


BULK_PATH = "input/2dRNA/group1/bulk_RawCounts.tsv"
SC_PATH = "input/2dRNA/group1/scRNA_CT2_top500_RawCounts.tsv"
SC_METADATA_PATH = "input/2dRNA/group1/scRNA_CT2_top500_Metadata.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pipeline_simple_dnn():
    # B is model input, C is model output
    B, C = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )

    # Hyperparam search for finding best SimpleDNN architecture
    param_grid = {
        "batch_size": [32, 64, 128],
        "lr": [0.001, 0.0001],
        "epochs": [1000],
        "patience": [30],
        "criterion": [nn.L1Loss(), nn.MSELoss()],
        "hidden_layers": [
            ([32], [0.1]),
            ([32, 16], [0.2, 0.1]),
            ([32, 32], [0.2, 0.1]),
            ([32, 32, 16], [0.2, 0.2, 0.1]),
            ([64], [0.1]),
            ([64, 16], [0.2, 0.1]),
            ([64, 32], [0.2, 0.1]),
            ([64, 32, 16], [0.2, 0.2, 0.1]),
            ([96], [0.1]),
            ([96, 32], [0.2, 0.1]),
            ([96, 64], [0.2, 0.1]),
            ([96, 64, 32], [0.2, 0.2, 0.1]),
            ([128], [0.2]),
            ([128, 32], [0.2, 0.1]),
            ([128, 64], [0.2, 0.1]),
            ([128, 64, 32], [0.2, 0.2, 0.1]),
            ([256], [0.2]),
            ([256, 32], [0.2, 0.1]),
            ([256, 128], [0.2, 0.1]),
            ([256, 128, 64], [0.2, 0.2, 0.1]),
        ],
    }

    best_params = hyperparam_search_simple_dnn(B, C, param_grid, k=3)

    X_train, X_test, Y_train, Y_test = train_test_split(B, C, test_size=0.2)

    final_model = SimpleDNN(
        best_params["input_dim"],
        best_params["output_dim"],
        best_params["hidden_layers"],
    ).to(DEVICE)

    print("\nTraining final model with best parameters...")
    train_simple_dnn(
        final_model,
        X_train,
        X_test,
        Y_train,
        Y_test,
        criterion=best_params["criterion"],
        batch_size=best_params["batch_size"],
        lr=best_params["lr"],
    )
    save_to_disk(final_model, X_test, Y_test, "SimpleDNN")
    print_eval(X_test, Y_test, model=final_model, name="SimpleDNN")


def pipeline_simple_linear():
    X_train, X_test, Y_train, Y_test = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )
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
    model = SimpleLinear(input_dim, output_dim)

    print("Training linear model...")
    train_simple_dnn(
        model,
        train_loader,
        test_loader,
        optimizer=optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4),
        criterion=nn.HuberLoss(delta=1.0),
        epochs=300,
    )
    print("Linear model training complete!")
    save_to_disk(model, X_test, Y_test, "SimpleLinear")
    print_eval(X_test, Y_test, model=model, name="SimpleLinear")


def pipeline_scaden():
    X_train, X_test, Y_train, Y_test = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # DNN sizes and dropout rates for each model in the Scaden ensemble
    architectures = {
        "m256": ([256, 128, 64, 32], [0, 0, 0, 0]),
        "m512": ([512, 256, 128, 64], [0, 0.3, 0.2, 0.1]),
        "m1024": ([1024, 512, 256, 128], [0, 0.6, 0.3, 0.1]),
    }
    ensemble_models = []
    for name, (hidden_layers, dropout_rates) in architectures.items():
        model = ScadenModel(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            hidden_units=hidden_layers,
            dropout_rates=dropout_rates,
        ).to(DEVICE)
        ensemble_models.append((name, model))

    train_scaden_ensemble(
        ensemble_models,
        train_loader,
        test_loader,
        lr=0.0001,
        criterion=nn.MSELoss(),
        epochs=5000,  # w/ early stopping
    )

    final_prediction = predict_ensemble(ensemble_models, test_loader)
    print_eval(X_test, Y_test, Y_pred=final_prediction, name="Scaden_Ensemble")


if __name__ == "__main__":
    if sys.argv[1] == "simplednn":
        pipeline_simple_dnn()
    elif sys.argv[1] == "simplelinear":
        pipeline_simple_linear()
    elif sys.argv[1] == "scaden":
        pipeline_scaden()
    else:
        raise ValueError("Invalid pipeline argument.")
