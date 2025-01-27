from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ray import tune
import heapq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(SimpleDNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim, dropout_rate in zip(*hidden_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SimpleLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinear, self).__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)


def train_simple_dnn(
    model: nn.Module,
    X_train,
    X_test,
    Y_train,
    Y_test,
    criterion=nn.L1Loss(),
    batch_size=32,
    epochs=1000,
    patience=20,
    lr=0.001,
):
    """
    Train a SimpleDNN PyTorch model on the given data.

    Args:
        model (nn.Module): PyTorch model to train.
        X_train (ndarray): Training input data.
        X_test (ndarray): Validation input data.
        Y_train (ndarray): Training output data.
        Y_test (ndarray): Validation output data.
        criterion (torch.nn): Loss function
        batch_size (int): Number of samples per batch.
        epochs (int)
        patience (int): Number of epochs to wait for improvement before early stopping.
        lr (float): Learning rate.

    Returns:
        float: Validation loss after training.
    """
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    for e in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)
                val_outputs = model(X_val)
                val_loss += criterion(val_outputs, Y_val).item()

        print(
            f"Epoch {e+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {e+1}. Best Val Loss: {best_val_loss:.4f}"
                )
                break

    return best_val_loss


def hyperparam_search_simple_dnn(X, Y, param_grid, k=5):
    """
    Hyperparameter search using k-fold CV to find the best DNN architecture.

    Args:
        X (ndarray): Input data.
        Y (ndarray): Output data.
        param_grid (dict): Dictionary of hyperparameters to search over.
        k (int): Number of folds for cross-validation.

    Returns:
        dict: Best hyperparameters found during search.

    Side Effects:
        - Prints average loss for each parameter set.
        - Prints best parameters and ranking.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    best_params = None
    best_combined_loss = float("inf")
    param_heap = []

    for idx, params in enumerate(ParameterGrid(param_grid)):
        print(f"\nParam Set: {idx+1}/{len(ParameterGrid(param_grid))}")

        fold_losses = []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            input_dim = X_train.shape[1]
            output_dim = Y_train.shape[1]

            model = SimpleDNN(input_dim, output_dim, params["hidden_layers"]).to(DEVICE)

            val_loss = train_simple_dnn(
                model,
                X_train,
                X_test,
                Y_train,
                Y_test,
                criterion=params["criterion"],
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                patience=params["patience"],
                lr=params["lr"],
            )
            fold_losses.append(val_loss)

        avg_loss = sum(fold_losses) / len(fold_losses)
        print(f"Avg Loss for Params {params}: {avg_loss:.4f}")

        heapq.heappush(param_heap, (avg_loss, params))

        if avg_loss < best_combined_loss:
            best_combined_loss = avg_loss
            best_params = params

    print(f"\nBest Params: {best_params}")

    print("\nParam Set Ranking:")
    ranked_params = sorted(param_heap)
    for rank, (loss, params) in enumerate(ranked_params, start=1):
        print(f"Rank {rank}: Loss = {loss:.4f}, Params = {params}")

    return best_params
