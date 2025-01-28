import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import warnings
from process import data_prep_pipeline
from SimpleDNN import (
    SimpleDNN,
    SimpleLinear,
)
from utils import print_eval

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


def load_weights(model: SimpleDNN, weights_path):
    """
    Load pretrained model weights from disk.
    """
    model.load_state_dict(torch.load(weights_path))
    model.to(DEVICE)


def predict_simple_dnn(model_path, hidden_layers):
    B, C = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )

    _, X_test, _, Y_test = train_test_split(B, C, test_size=0.2)
    input_dim = X_test.shape[1]
    output_dim = Y_test.shape[1]

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    model = SimpleDNN(input_dim, output_dim, hidden_layers)
    load_weights(model, model_path)

    print_eval(Y_test, model=model, X_test=X_test, name="SimpleDNN")


def predict_ensemble_avg(models, X_test):
    print("\nEnsemble predictions (average of 3 DNNs)...")
    predictions = np.zeros((X_test.shape[0], 11))
    for _, model in models:
        model.eval()
        with torch.no_grad():
            X_test = torch.tensor(X_test, dtype=torch.float32)
            predictions += model(X_test.to(DEVICE)).cpu().numpy()
    predictions /= len(models)
    return predictions


def predict_scaden(m256_path, m512_path, m1024_path):
    B, C = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )

    _, X_test, _, Y_test = train_test_split(B, C, test_size=0.2)
    input_dim = X_test.shape[1]
    output_dim = Y_test.shape[1]

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    m256 = SimpleDNN(input_dim, output_dim, ([256, 128, 64, 32], [0, 0, 0, 0]))
    m512 = SimpleDNN(input_dim, output_dim, ([512, 256, 128, 64], [0, 0.3, 0.2, 0.1]))
    m1024 = SimpleDNN(
        input_dim, output_dim, ([1024, 512, 256, 128], [0, 0.6, 0.3, 0.1])
    )
    load_weights(m256, m256_path)
    load_weights(m512, m512_path)
    load_weights(m1024, m1024_path)

    Y_pred = predict_ensemble_avg(
        [("m256", m256), ("m512", m512), ("m1024", m1024)], X_test
    )

    print_eval(Y_test, Y_pred, name="ScadenEnsemble")


def main():
    # predict_simple_dnn(
    #     model_path="output/2dRNA/20250127_1614/m256.pth",
    #     hidden_layers=([256, 128, 64, 32], [0, 0, 0, 0]),
    # )
    predict_scaden(
        m256_path="output/2dRNA/20250127_1614/m256.pth",
        m512_path="output/2dRNA/20250127_1619/m512.pth",
        m1024_path="output/2dRNA/20250127_1620/m1024.pth",
    )


if __name__ == "__main__":
    main()
