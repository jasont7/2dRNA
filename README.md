# **Cellular Deconvolution Project Outline [CSC 427]**

**Reimplementing: Scaden** (Menden et al., _Science Advances_ 2020)

## **Goal**

Deconvolution of bulk RNA-seq data: transform tissue-level gene expression profiles (GEPs) to cell-level GEPs by learning a non-linear relationship between bulk and single-cell (scRNA-seq) data samples.

## **Terminology**

### S: Single-cell matrix

-   **Dimensions:** c cells × p genes (c: ~5000, p: ~200) for one patient
-   **Structure:** Multiple patients stacked vertically: \[n patients × c cells\] × p genes
-   Expensive to generate (requires cellular-level sampling)
-   AKA GEP matrix at the _cell_ level

### B: Bulk matrix

-   **Dimensions:** p genes × n patients
-   **Structure:** Each entry represents GEP for an entire tissue (no cell-specific information)
-   **Cost:** Cheap to generate (high-level/abstract view)

### R: Reference matrix (derived from **S** or simulated)

-   **Dimensions:** p genes × k cell types (k < 20)
-   **Structure:** Aggregated (averaged) GEPs by cell type from **S**
-   AKA GEP matrix at the _cell-type_ level

### X: Cell-Type Abundance matrix

-   **Dimensions:** k cell types × n patients
-   **Structure:** Each entry represents the abundance of each cell type in **R**
-   **Operation:** R × X = pseudo-bulk (like **B** but adjusted for sequencing and cellular detail)

### Relationship: B ≈ R × X

-   Since **B** and **S** samples are derived using different technologies, B = XR does **not** hold.
-   **Goal:** Learn a non-linear relationship from **B** to **X'**.
-   **Outcome:** Estimated **X'** can be combined with any **R** (real, simulated, or generic) to produce sc-adjusted pseudo-bulk for more accurate GEPs than the input bulk.

## **Methods**

-   **Build a non-linear deep learning (DL) model** to learn the relationship between bulk samples and cell-type abundance fractions.

-   **Datasets**:
    -   **PBMC ~10k cells**: [Filtered feature barcode matrix (HDF5)](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-2-0-0)
    -   **PBMC ~3k cells**: [Filtered feature barcode matrix (HDF5)](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-3-k-1-standard-2-0-0)

## **Related Works**

### **1. Scaden** (Menden et al., _Science Advances_ 2020)

-   **Approach:** Non-linear deep learning model to predict cell-type proportions from bulk RNA-seq using single-cell data.
-   **Outcome:** More accurate deconvolution than linear models when trained on large, representative datasets.
-   **Methods:** Neural networks trained on synthetic bulk samples created from single-cell datasets.
-   **Evaluation:** Compares predicted cell-type abundance fractions (**X'**) against ground-truth cell-type proportions derived from paired bulk and single-cell RNA-seq datasets (e.g., PBMC datasets). Metrics include **Mean Absolute Error (MAE)** and **correlation** between predicted and true proportions.

### **2. SQUID** (Cobos et al., _Genome Biology_ 2023)

-   **Approach:** Benchmarks methods to deconvolve bulk RNA-seq into cell-type contributions; introduces SQUID.
-   **Outcome:** Strengths, limitations, and applicability of methods across tissues/conditions, and evaluation of SQUID's performance.
-   **Methods:** Compares bulk to single-cell reference profiles, estimating the fraction of each cell type.
-   **Comparison:** SQUID is considered the current SOTA linear deconvolution algorithm. Our approach focuses on a **non-linear model** to learn more complex relationships.

### **3. DSSC** (Wang et al., _BMC Genomics_ 2024)

-   **Approach:** Deconvolution algorithm to infer both cell-type proportions and GEPs from bulk samples.
-   **Outcome:** Simultaneously infers cell-type proportions and GEPs for various datasets.
-   **Methods:** Linear model leveraging gene-gene and sample-sample patterns in bulk/single-cell data.
-   **Comparison:** DSSC uses a **linear algorithm**. Our approach uses a **non-linear model** to bypass complex data pipelines and capture deeper relationships.

### **4. SCDC** (Dong et al., 2021)

-   **Approach:** Aggregates multiple single-cell datasets to create a more robust reference for deconvolution in heterogeneous tissues.
-   **Outcome:** Outperforms existing methods in decomposition accuracy and returns more biologically relevant results.
-   **Methods:** Linear, multi-subject, ensemble-based deconvolution using multiple single-cell datasets.
-   **Comparison:** SCDC focuses on reference aggregation; our approach develops a **non-linear tool** that avoids explicit batch effect correction.

### **5. MuSiC** (Wang et al., _Nature Communications_ 2019)

-   **Approach:** Similar to CIBERSORTx but incorporates subject-level variability in deconvolution.
-   **Methods:** Weighted non-negative least squares (NNLS) with single-cell references, using subject-specific weights.
-   **Comparison:** MuSiC relies on a **weighted linear model**, while our approach focuses on learning a **non-linear model**.

### **6. CIBERSORTx** (Newman et al., _Nature Biotech_ 2019)

-   **Approach:** Predicts cell-type proportions from bulk RNA-seq and adjusts bulk GEPs accordingly.
-   **Outcome:** Provides cell-type proportions and adjusted bulk GEPs.
-   **Methods:** Linear deconvolution and reference profiles from single-cell data, using batch correction and smoothing.
-   **Comparison:** CIBERSORTx uses a **linear method** and signature gene sets; our approach learns a **non-linear relationship**.

### **7. Computational Deconvolution of Transcriptomics Data from Mixed Cell Populations** (Cobos et al., _Oxford Bioinformatics_ 2018)

-   **Approach:** Review of cell-type deconvolution methods for bulk RNA-seq samples.
-   **Outcome:** Provides a foundational understanding of cellular deconvolution and how single-cell data informs bulk deconvolution.
