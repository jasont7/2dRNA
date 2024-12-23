import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models, losses, optimizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance


# FEAT GENE SELECTION: differential expression analysis using MAST (Placeholder)
def perform_DE_analysis(single_cell_data):
    # Placeholder function: In practice, use a DE analysis package compatible with Python or import results
    # For demonstration, we'll select top N genes with highest variance
    variances = np.var(single_cell_data, axis=0)
    top_indices = np.argsort(variances)[-500:]  # Select top 500 genes
    return top_indices


# FEAT GENE SELECTION: Random Forest feature selection
def random_forest_feature_selection(single_cell_data, cell_type_labels, gene_indices):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(single_cell_data[:, gene_indices], cell_type_labels)
    importances = rf.feature_importances_
    top_indices = gene_indices[np.argsort(importances)[-200:]]  # Select top 200 genes
    return top_indices


# DATASET AUGMENTATION: generate pseudo-bulk samples from single-cell data
def generate_pseudo_bulk_samples(
    single_cell_data, cell_type_labels, num_samples=10, subset_fraction=0.7
):
    pseudo_bulk_data = []
    pseudo_bulk_labels = []
    num_cells = single_cell_data.shape[0]
    subset_size = int(num_cells * subset_fraction)
    for _ in range(num_samples):
        indices = np.random.choice(num_cells, subset_size, replace=False)
        subset_data = single_cell_data[indices]
        subset_labels = cell_type_labels[indices]
        bulk_expression = np.sum(subset_data, axis=0)
        cell_type_counts = np.bincount(
            subset_labels, minlength=np.max(cell_type_labels) + 1
        )
        cell_type_proportions = cell_type_counts / np.sum(cell_type_counts)
        pseudo_bulk_data.append(bulk_expression)
        pseudo_bulk_labels.append(cell_type_proportions)
    return np.array(pseudo_bulk_data), np.array(pseudo_bulk_labels)


# DATASET AUGMENTATION: add negative binomial noise to bulk data
def add_negative_binomial_noise(bulk_data):
    noisy_bulk_data = np.random.negative_binomial(n=bulk_data, p=0.5)
    return noisy_bulk_data


# Custom loss functions
def kl_divergence_loss(y_true, y_pred):
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1)
    y_true = tf.clip_by_value(y_true, epsilon, 1)
    return tf.reduce_sum(y_true * tf.math.log(y_true / y_pred), axis=-1)


def wasserstein_loss(y_true, y_pred):
    return tf.py_function(wasserstein_distance, [y_true, y_pred], tf.float32)


def aitchison_distance_loss(y_true, y_pred):
    epsilon = 1e-8
    y_true = tf.clip_by_value(y_true, epsilon, 1)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1)
    log_ratio_diff = tf.math.log(
        y_true / tf.reduce_prod(y_true, axis=-1, keepdims=True)
    ) - tf.math.log(y_pred / tf.reduce_prod(y_pred, axis=-1, keepdims=True))
    return tf.sqrt(tf.reduce_mean(tf.square(log_ratio_diff), axis=-1))


def huber_loss(y_true, y_pred, delta=0.1):
    return tf.keras.losses.Huber(delta=delta)(y_true, y_pred)


def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = 1e-8
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    ce = -y_true * tf.math.log(y_pred)
    weight = tf.math.pow(1 - y_pred, gamma)
    fl = weight * ce
    return tf.reduce_sum(fl, axis=-1)


def build_model(input_dim, output_dim):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_dim, activation="softmax"))
    return model


def main(data_augmentation_option, loss_function_option):
    # Load and preprocess data
    # Placeholder: Replace with actual data loading
    # Assume single_cell_data: NumPy array of shape (num_cells, num_genes)
    # Assume cell_type_labels: NumPy array of shape (num_cells,)
    # Assume bulk_data: NumPy array of shape (num_samples, num_genes)
    # Assume bulk_labels: NumPy array of shape (num_samples, num_cell_types)

    # For demonstration purposes, generate synthetic data
    num_cells = 10000
    num_genes = 1000
    num_cell_types = 5
    num_bulk_samples = 300

    np.random.seed(42)
    single_cell_data = np.random.poisson(lam=5, size=(num_cells, num_genes))
    cell_type_labels = np.random.randint(0, num_cell_types, size=(num_cells,))

    # Perform DE analysis and Random Forest feature selection
    de_gene_indices = perform_DE_analysis(single_cell_data)
    selected_gene_indices = random_forest_feature_selection(
        single_cell_data, cell_type_labels, de_gene_indices
    )

    # Reduce single-cell data to selected genes
    single_cell_data = single_cell_data[:, selected_gene_indices]
    num_selected_genes = len(selected_gene_indices)

    # Generate bulk data (Placeholder)
    bulk_data = np.random.poisson(lam=50, size=(num_bulk_samples, num_selected_genes))
    bulk_labels = np.random.dirichlet(
        alpha=np.ones(num_cell_types), size=num_bulk_samples
    )

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        bulk_data, bulk_labels, test_size=0.2, random_state=42
    )

    # # Data Augmentation
    # if data_augmentation_option == "augment_X":
    #     # Augment X only by random sampling S
    #     augmented_data, augmented_labels = generate_pseudo_bulk_samples(
    #         single_cell_data, cell_type_labels
    #     )
    #     X_train = np.vstack([X_train, augmented_data])
    #     y_train = np.vstack([y_train, augmented_labels])
    # elif data_augmentation_option == "augment_B":
    #     # Augment B by introducing noise
    #     noisy_data = add_negative_binomial_noise(X_train)
    #     X_train = np.vstack([X_train, noisy_data])
    #     y_train = np.vstack([y_train, y_train])  # Duplicate labels
    # elif data_augmentation_option == "augment_both":
    #     # Augment both S and B
    #     augmented_data1, augmented_labels1 = generate_pseudo_bulk_samples(
    #         single_cell_data, cell_type_labels
    #     )
    #     noisy_data = add_negative_binomial_noise(X_train)
    #     X_train = np.vstack([X_train, augmented_data1, noisy_data])
    #     y_train = np.vstack([y_train, augmented_labels1, y_train])
    # # No augmentation for 'no_augmentation' option

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Build the model
    model = build_model(input_dim=num_selected_genes, output_dim=num_cell_types)

    # Select loss function
    if loss_function_option == "standard":
        loss_fn = "categorical_crossentropy"
    elif loss_function_option == "standard_plus_kl":

        def custom_loss(y_true, y_pred):
            return losses.categorical_crossentropy(y_true, y_pred) + kl_divergence_loss(
                y_true, y_pred
            )

        loss_fn = custom_loss
    elif loss_function_option == "kl":
        loss_fn = kl_divergence_loss
    elif loss_function_option == "wasserstein":
        loss_fn = wasserstein_loss
    elif loss_function_option == "aitchison":
        loss_fn = aitchison_distance_loss
    elif loss_function_option == "huber":
        loss_fn = huber_loss
    elif loss_function_option == "focal":
        loss_fn = focal_loss
    else:
        raise ValueError("Invalid loss function option")

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
    )

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    # Save the model
    model.save("cell_type_abundance_model.h5")


if __name__ == "__main__":
    # User options
    data_augmentation_options = [
        "no_augmentation",
        "augment_X",
        "augment_B",
        "augment_both",
    ]
    loss_function_options = [
        "standard",
        "standard_plus_kl",
        "kl",
        "wasserstein",
        "aitchison",
        "huber",
        "focal",
    ]

    # Example usage
    data_augmentation_option = "augment_both"  # Replace with user input
    loss_function_option = "standard_plus_kl"  # Replace with user input

    main(data_augmentation_option, loss_function_option)
