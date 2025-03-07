import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import torch
import warnings
from process import data_prep_pipeline
from SimpleDNN import (
    SimpleDNN,
)
from utils import eval_metrics

warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=120)
np.set_printoptions(precision=4, suppress=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BULK_PATH = "input/2dRNA/bulk_RawCounts.tsv"
SC_PATH = "input/2dRNA/scRNA_CT2_top500_RawCounts.tsv"
SC_METADATA_PATH = "input/2dRNA/scRNA_CT2_top500_Metadata.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(model: SimpleDNN, weights_path):
    """
    Load pretrained model weights from disk.
    """
    model.load_state_dict(torch.load(weights_path))
    model.to(DEVICE)


def predict_simple_dnn(model_path, hidden_layers, n_splits=5):
    B, C = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )
    scaler = StandardScaler()
    B = scaler.fit_transform(B)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    eval_results = []

    for train_idx, test_idx in kfold.split(B):
        _, X_test = B[train_idx], B[test_idx]
        _, Y_test = C[train_idx], C[test_idx]

        model = SimpleDNN(
            input_dim=B.shape[1], output_dim=C.shape[1], hidden_layers=hidden_layers
        )
        load_weights(model, model_path)

        eval_result = eval_metrics(Y_test, model=model, X_test=X_test)
        eval_results.append(eval_result)

    avg_eval_result = {
        "target_mean": np.mean([result["target_mean"] for result in eval_results]),
        "target_var": np.mean([result["target_var"] for result in eval_results]),
        "mse": np.mean([result["mse"] for result in eval_results]),
        "r2": np.mean([result["r2"] for result in eval_results]),
        "kl_div": np.mean([result["kl_div"] for result in eval_results]),
        "wasserstein": np.mean([result["wasserstein"] for result in eval_results]),
    }

    print(f"\nAverage Evaluation Metrics Across K={n_splits} Folds:")
    print(f" - Target value mean: {avg_eval_result['target_mean']:.7f}")
    print(f" - Target value variance: {avg_eval_result['target_var']:.7f}")
    print(f" - MSE: {avg_eval_result['mse']:.7f}")
    print(f" - R²: {avg_eval_result['r2']:.7f}")
    print(f" - KL Divergence: {avg_eval_result['kl_div']:.7f}")
    print(f" - Wasserstein Distance: {avg_eval_result['wasserstein']:.7f}")


def predict_ensemble_avg(models, X_test):
    predictions = np.zeros((X_test.shape[0], 11))
    for _, model in models:
        model.eval()
        with torch.no_grad():
            X_test = torch.tensor(X_test, dtype=torch.float32)
            predictions += model(X_test.to(DEVICE)).cpu().numpy()
    predictions /= len(models)
    return predictions


def predict_scaden(m256_path, m512_path, m1024_path, n_splits=5):
    B, C = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )
    scaler = StandardScaler()
    B = scaler.fit_transform(B)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    eval_results = []

    for train_idx, test_idx in kfold.split(B):
        _, X_test = B[train_idx], B[test_idx]
        _, Y_test = C[train_idx], C[test_idx]

        m256 = SimpleDNN(
            input_dim=B.shape[1],
            output_dim=C.shape[1],
            hidden_layers=([256, 128, 64, 32], [0, 0, 0, 0]),
        )
        m512 = SimpleDNN(
            input_dim=B.shape[1],
            output_dim=C.shape[1],
            hidden_layers=([512, 256, 128, 64], [0, 0.3, 0.2, 0.1]),
        )
        m1024 = SimpleDNN(
            input_dim=B.shape[1],
            output_dim=C.shape[1],
            hidden_layers=([1024, 512, 256, 128], [0, 0.6, 0.3, 0.1]),
        )
        load_weights(m256, m256_path)
        load_weights(m512, m512_path)
        load_weights(m1024, m1024_path)

        Y_pred = predict_ensemble_avg(
            [("m256", m256), ("m512", m512), ("m1024", m1024)], X_test
        )
        eval_result = eval_metrics(Y_test, Y_pred)
        eval_results.append(eval_result)

    avg_eval_result = {
        "target_mean": np.mean([result["target_mean"] for result in eval_results]),
        "target_var": np.mean([result["target_var"] for result in eval_results]),
        "mse": np.mean([result["mse"] for result in eval_results]),
        "r2": np.mean([result["r2"] for result in eval_results]),
        "kl_div": np.mean([result["kl_div"] for result in eval_results]),
        "wasserstein": np.mean([result["wasserstein"] for result in eval_results]),
    }

    print(f"\nAverage Evaluation Metrics Across K={n_splits} Folds:")
    print(f" - Target value mean: {avg_eval_result['target_mean']:.7f}")
    print(f" - Target value variance: {avg_eval_result['target_var']:.7f}")
    print(f" - MSE: {avg_eval_result['mse']:.7f}")
    print(f" - R²: {avg_eval_result['r2']:.7f}")
    print(f" - KL Divergence: {avg_eval_result['kl_div']:.7f}")
    print(f" - Wasserstein Distance: {avg_eval_result['wasserstein']:.7f}")


def main():
    # predict_simple_dnn(
    #     model_path="output/2dRNA/20250128_0224/SimpleDNN.pth",
    #     hidden_layers=([1024, 512], [0.05, 0.05]),
    # )
    predict_scaden(
        m256_path="output/2dRNA/20250127_1614/m256.pth",
        m512_path="output/2dRNA/20250127_1619/m512.pth",
        m1024_path="output/2dRNA/20250127_1620/m1024.pth",
    )


if __name__ == "__main__":
    main()
