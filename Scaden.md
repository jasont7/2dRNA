## Scaden Ensemble Overview

Scaden utilizes an ensemble of three DNN models (`m256`, `m512`, and `m1024`) with the following differences:

-   **Layer sizes**: Each model has a unique architecture defined by the number of neurons in each hidden layer.
-   **Dropout rates**: Each model uses specific dropout rates for regularization, applied after each hidden layer.
-   **Final prediction**: The predictions from the three models are averaged to provide the final cell-type fraction estimation.

---

### Model Architectures

The three architectures (`m256`, `m512`, `m1024`) are as follows:

-   **`m256`**:

    -   Hidden layers: `[256, 128, 64, 32]`
    -   Dropout rates: `[0, 0, 0, 0]`

-   **`m512`**:

    -   Hidden layers: `[512, 256, 128, 64]`
    -   Dropout rates: `[0, 0.3, 0.2, 0.1]`

-   **`m1024`**:
    -   Hidden layers: `[1024, 512, 256, 128]`
    -   Dropout rates: `[0, 0.6, 0.3, 0.1]`

---

### Loss, Optimizer, and Training

-   **Loss Function**: L1 loss (mean absolute error) is used for training.
-   **Optimizer**: Adam optimizer with a learning rate of `0.0001`.
-   **Training Parameters**:
    -   **Batch size**: `128`
    -   **Epochs**: Up to `5000`, with **early stopping** to terminate training if validation loss does not improve for a set number of epochs.
-   **Output**: Softmax activation at the output layer ensures predictions represent probabilities (cell type fractions).

---

### Final Prediction

The outputs of the three models are averaged to provide the final prediction:

Final Prediction = 1/3 \* Σ(Pred_i for i=1 to 3)
