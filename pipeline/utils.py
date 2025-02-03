import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy, wasserstein_distance
import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_to_disk(model, X_test=None, Y_test=None, name="Model"):
    """
    Save model, predictions, and true fractions to disk.
    """
    os.makedirs("output", exist_ok=True)
    dtnum = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    model_dir = os.path.join("output", "2dRNA", dtnum)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {name} to {model_path}")

    if X_test is not None and Y_test is not None:
        # Save predictions and true fractions
        X_test = torch.tensor(X_test, dtype=torch.float32)
        Y_test = torch.tensor(Y_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            predictions = model(X_test.to(DEVICE)).cpu().numpy()
        preds_file = os.path.join(model_dir, f"{name}_pred_fractions.csv")
        true_fractions_file = os.path.join(model_dir, f"{name}_true_fractions.csv")
        np.savetxt(preds_file, predictions, delimiter=",")
        np.savetxt(true_fractions_file, Y_test.numpy(), delimiter=",")
        print(f"Saved predictions to {preds_file}")
        print(f"Saved true fractions to {true_fractions_file}")


def eval_metrics(Y_test, Y_pred=None, model=None, X_test=None):
    """
    Print evaluation metrics for model on test set by comparing Y_test to Y_pred.
    Computes MSE, R², KL Divergence, and Wasserstein Loss.
    Y_pred is the model's predictions on X_test IF a model is provided. Otherwise,
    Y_pred must be provided to be compared directly.
    """
    if model is None and Y_pred is None:
        raise ValueError("Either model or Y_pred must be provided.")

    if not isinstance(Y_test, torch.Tensor):
        Y_test = torch.tensor(Y_test, dtype=torch.float32)

    if model is not None:
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            Y_pred = model(X_test.to(Y_test.device)).cpu()

    if not isinstance(Y_pred, torch.Tensor):
        Y_pred = torch.tensor(Y_pred, dtype=torch.float32)

    target_mean = Y_test.mean().item()
    target_var = torch.var(Y_test).item()

    mse = torch.mean((Y_pred - Y_test) ** 2).item()

    # R-squared (R²)
    ss_total = torch.sum((Y_test - target_mean) ** 2).item()
    ss_residual = torch.sum((Y_test - Y_pred) ** 2).item()
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else float("nan")

    # KL Divergence (requires probability distributions, so normalize)
    Y_test_probs = F.softmax(Y_test, dim=1).numpy()
    Y_pred_probs = F.softmax(Y_pred, dim=1).numpy()
    kl_div = entropy(Y_test_probs.T, Y_pred_probs.T).mean()

    wasserstein = (
        sum(
            wasserstein_distance(Y_test[i].numpy(), Y_pred[i].numpy())
            for i in range(Y_test.shape[0])
        )
        / Y_test.shape[0]
    )

    return {
        "target_mean": target_mean,
        "target_var": target_var,
        "mse": mse,
        "r2": r2,
        "kl_div": kl_div,
        "wasserstein": wasserstein,
    }
