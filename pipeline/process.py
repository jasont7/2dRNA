import pandas as pd
import numpy as np


def load_data(bulk_path, sc_path, sc_metadata_path):
    bulk_df = pd.read_csv(bulk_path, sep="\t")
    sc_df = pd.read_csv(sc_path, sep="\t")
    sc_metadata_df = pd.read_csv(sc_metadata_path, sep="\t")

    print("B Matrix (Tissue GEPs) Sample:\n")
    print(bulk_df.iloc[:, :6].head(5))
    print(
        f"\nB DIMENSIONS: rows (genes) = {bulk_df.shape[0]}, columns (patients) = {bulk_df.shape[1]}"
    )
    print("\n----------------------------------------------\n")

    print("S Matrix (Cell GEPs) Sample:\n")
    print(sc_df.iloc[:, :12].head(5))
    print(
        f"\nS DIMENSIONS: rows (patients x cells) = {sc_df.shape[0]}, columns (genes) = {sc_df.shape[1]}"
    )
    print("\n----------------------------------------------\n")

    print("S Metadata Matrix Sample:\n")
    print(sc_metadata_df.head(5))
    print(
        f"\nS METADATA DIMENSIONS: rows (patients x cells) = {sc_metadata_df.shape[0]}, columns (metadata) = {sc_metadata_df.shape[1]}"
    )
    print("\n----------------------------------------------\n")

    return bulk_df, sc_df, sc_metadata_df


def process_bulk(bulk: pd.DataFrame, sc: pd.DataFrame, sc_metadata: pd.DataFrame):
    """
    Process bulk data to keep only common genes with single-cell data, and
    assert that all bulk samples have a corresponding single-cell sample.

    Args:
        bulk (pd.DataFrame): Bulk GEP data (genes x patients).
        sc (pd.DataFrame): Single-cell GEP data ((patients * cells) x genes).
        sc_metadata (pd.DataFrame): Single-cell metadata ((patients * cells) x metadata).

    Returns:
        tuple: Processed B matrix (patients x genes) and patient IDs.
    """

    # Filter B to keep only common genes with S
    bulk_genes_all = bulk["gene_symbol"].str.strip().str.lower()
    common_genes = set(bulk_genes_all).intersection(
        sc.columns[2:].str.strip().str.lower()
    )
    filtered_bulk = bulk[bulk_genes_all.isin(common_genes)].drop_duplicates(
        subset="gene_symbol", keep="first"
    )
    filtered_bulk_vals = filtered_bulk.iloc[:, 2:]  # drop gene_id and gene_symbol cols

    # Normalize and convert to np array
    B = np.log1p(filtered_bulk_vals.values.T)
    print(f"Filtered B dims (patients x genes): {B.shape}\n")

    # Assert that patient IDs in S match B
    sc_patient_ids = sc_metadata["patient_id"].unique()
    bulk_patient_ids = filtered_bulk_vals.columns
    if not all(i in bulk_patient_ids for i in sc_patient_ids):
        raise ValueError("Patient IDs in S do not match B. Check mapping.")

    return B, bulk_patient_ids


def process_single_cell(sc_metadata: pd.DataFrame, pids: np.ndarray, n_aug, aug_ratio):
    """
    Derive augmented C matrices with random sampling (not stratified).

    Args:
        sc_metadata (pd.DataFrame): Single-cell metadata containing patient and cell type information.
        pids (np.ndarray): Array of patient IDs in bulk data.
        n_aug (int): Number of augmentations per patient.
        aug_ratio (float): Fraction of cells to sample for each augmentation (e.g., 0.9 for 90%).

    Returns:
        np.ndarray: Augmented C matrices (patients x n_augs x cell types).
        list: Successfully processed patient IDs.
    """
    ct_labels = sc_metadata["cell_type_1"].dropna().unique()
    C_augs = []  # Augmentations for all patients
    processed_patients = []  # Successfully processed patient IDs

    for pid in pids:
        print(f"Augmenting Patient {pid}")
        patient_cells = sc_metadata[sc_metadata["patient_id"] == pid]
        if patient_cells.empty:
            print(f"  Skipping (no cells)")
            continue

        patient_augs = []

        for _ in range(n_aug):
            # Randomly sample a subset of all cells
            n_sample = max(1, int(len(patient_cells) * aug_ratio))
            sampled_cells = patient_cells.sample(
                n=n_sample, replace=False, random_state=None
            )

            # Calculate cell type fractions for this augmentation
            ct_fractions = sampled_cells["cell_type_1"].value_counts(normalize=True)
            all_ct_fractions = {ct: ct_fractions.get(ct, 0.0) for ct in ct_labels}
            patient_augs.append(list(all_ct_fractions.values()))

        C_augs.append(patient_augs)
        processed_patients.append(pid)

    print(f"\nProcessed patients: {len(processed_patients)}")

    C_augs = np.array(C_augs)
    # Flatten to 2D so each row is an augmentation for a specific patient
    C_flat = C_augs.reshape(-1, C_augs.shape[2])
    print(f"C dims ((patients * n_augs) x CTs): {C_flat.shape}")

    return C_flat, processed_patients


def data_prep_pipeline(bulk_path, sc_path, sc_metadata_path, n_aug, aug_ratio):
    """
    Pipeline for preparing all necessary data for training.

    Args:
        bulk_path (str): Path to bulk GEP data.
        sc_path (str): Path to single-cell GEP data.
        sc_metadata_path (str): Path to single-cell metadata.
        n_aug (int): Number of augmentations per patient.
        aug_ratio (float): Fraction of cells to sample for each augmentation (e.g., 0.9 for 90%).

    Returns:
        tuple: (B, C) data for training.
    """

    bulk_df, sc_df, sc_metadata_df = load_data(bulk_path, sc_path, sc_metadata_path)
    B, patient_ids = process_bulk(bulk_df, sc_df, sc_metadata_df)
    C_flat, processed_patient_ids = process_single_cell(
        sc_metadata_df, patient_ids, n_aug, aug_ratio
    )

    # Keep only patients with both B and C data
    B_filtered = B[np.isin(patient_ids, processed_patient_ids)]
    print(f"\nFiltered B dims (patients x genes): {B_filtered.shape}")

    # Repeat B_i for each augmentation of S_i
    B_aug = np.repeat(B_filtered, n_aug, axis=0)
    print(f"Flattened B dims ((patients * n_augs) x genes): {B_aug.shape}")

    return B_aug, C_flat
