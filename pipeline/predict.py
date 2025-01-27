import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import warnings
from process import data_prep_pipeline
from SimpleDNN import (
    SimpleDNN,
    SimpleLinear,
)
from Scaden import ScadenModel, predict_ensemble
from utils import print_eval

warnings.filterwarnings("ignore")
np.set_printoptions(linewidth=120)
np.set_printoptions(precision=4, suppress=True)


BULK_PATH = "input/2dRNA/group1/bulk_RawCounts.tsv"
SC_PATH = "input/2dRNA/group1/scRNA_CT2_top500_RawCounts.tsv"
SC_METADATA_PATH = "input/2dRNA/group1/scRNA_CT2_top500_Metadata.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_predict(model, weights_path, X_test):
    """
    Load pretrained model and make predictions on X_test.
    """
    model.load_state_dict(torch.load(weights_path))
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(X_test.to(DEVICE)).cpu().numpy()
    return predictions


def main():
    """
    Load dataset and pretrained model to make predictions and print eval metrics.
    """
    X_train, X_test, Y_train, Y_test = data_prep_pipeline(
        BULK_PATH, SC_PATH, SC_METADATA_PATH, n_aug=30, aug_ratio=0.9
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Example (replace with your model)
    model = SimpleDNN(1685, 11, ([96, 64], [0.2, 0.1]))

    predictions = load_and_predict(
        model, "output/2dRNA/20250126_2254/SimpleDNN.pth", X_test
    )

    print_eval(X_test, Y_test, predictions, name="SimpleDNN")


if __name__ == "__main__":
    main()
