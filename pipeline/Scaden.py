import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import save_to_disk

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScadenModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rates):
        super(ScadenModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_units):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rates[i]),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return nn.functional.softmax(self.model(x), dim=1)


def train_scaden_ensemble(
    models, train_loader, val_loader, lr, criterion, epochs, patience=10
):
    best_models = []
    for name, model in models:
        print(f"Training Scaden sub-model {name}...")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_val_loss = float("inf")
        early_stop_counter = 0
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

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_val, Y_val in val_loader:
                    X_val, Y_val = X_val.to(DEVICE), Y_val.to(DEVICE)
                    val_outputs = model(X_val)
                    val_loss += criterion(val_outputs, Y_val).item()
            print(
                f"Epoch {e+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict()
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        model.load_state_dict(best_model_state)
        best_models.append(model)
        save_to_disk(model, name=f"Scaden_{name}")

    return best_models


def predict_ensemble(models, test_loader):
    predictions = []
    for name, model in models:
        model.eval()
        model_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(DEVICE)
                model_preds.append(model(X_batch).cpu().numpy())
        predictions.append(np.vstack(model_preds))

    # Average predictions from all models
    final_prediction = np.mean(predictions, axis=0)
    return final_prediction
