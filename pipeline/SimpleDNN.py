from sklearn.model_selection import KFold, ParameterGrid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import heapq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate):
        super(SimpleDNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for layer_size in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, layer_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = layer_size
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


def train_nn_model(model, train_loader, val_loader, optimizer, criterion, epochs):
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
    return val_loss


def hyperparam_search_simple_dnn(model_class, param_grid, X, Y, k=5):
    """
    Hyperparameter search using k-fold CV to find the best DNN architecture.
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    best_params = None
    best_combined_loss = float("inf")
    param_heap = []

    for idx, params in enumerate(ParameterGrid(param_grid)):
        print(f"\nParam Set: {idx+1}/{len(ParameterGrid(param_grid))}")
        model_params = {
            k: v for k, v in params.items() if k not in ["lr", "epochs", "criterion"]
        }

        fold_losses = []
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(Y_train, dtype=torch.float32),
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(Y_val, dtype=torch.float32),
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            model = model_class(**model_params).to(DEVICE)
            val_loss = train_nn_model(
                model,
                train_loader,
                val_loader,
                optimizer=optim.Adam(model.parameters(), params["lr"]),
                criterion=params["criterion"],
                epochs=params["epochs"],
            )
            fold_losses.append(val_loss)

        avg_loss = sum(fold_losses) / len(fold_losses)
        print(f"Avg Loss for Params {params}: {avg_loss:.4f}")

        heapq.heappush(param_heap, (avg_loss, params))

        # Update best params
        if avg_loss < best_combined_loss:
            best_combined_loss = avg_loss
            best_params = params

    print(f"\nBest Params: {best_params} with Loss: {best_combined_loss:.4f}")

    print("\nRanked Parameter Sets:")
    ranked_params = sorted(param_heap)
    for rank, (loss, param_set) in enumerate(ranked_params, start=1):
        print(f"Rank {rank}: Loss = {loss:.4f}, Params = {model_params}")

    return best_params
