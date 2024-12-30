# **2dRNA: Deep Deconvolution of RNA-seq Data**

## **Overview**

**Bulk RNA-seq** measures gene expression across entire tissues but does not provide cell-specific information. Instead, it captures an aggregate signal from all present cell types without distinguishing their individual contributions. **Cellular deconvolution** addresses this limitation by breaking down the bulk signal into cell type contributions by estimating the fraction of each cell type within a sample, providing deeper insights into its composition.

We aim to bridge the gap between bulk (which is affordable but lacks cell-specific detail) and single-cell (which provides detailed cell-level information but is expensive and difficult to scale) data by learning a **non-linear** relationship between them using **paired samples** (i.e., bulk and single-cell data from the _same_ patients/tissues).

### Why Is This Important?

-   Many diseases, such as cancer or autoimmune disorders, involve changes in the **proportion of cell types** within tissues. Understanding these changes can inform diagnosis and treatment.
-   Bulk RNA-seq is **affordable** and widely available, making deconvolution a practical tool to use with existing methods and datasets.
-   By building a model that estimates cell type proportions given only an unseen bulk sample, it can allow us to **"bypass"** the need for an expensive single-cell sample.

### Key Concepts for Non-Biologists

-   **Gene Expression**: The process by which a gene's information is used to produce functional molecules (e.g., proteins). It’s a measure of how "active" a gene is in a particular sample.

    -   **Example**: Measuring gene expression helps identify which genes are turned on or off in a disease state compared to healthy cells.

-   **RNA-seq** (RNA Sequencing): A technology used to quantify gene expression levels. This can be applied at **two scales**:

    1. **Single-Cell RNA-seq** (scRNA-seq): Measures gene expression in _individual cells_, providing a detailed view of cell composition.

        - **Example**: scRNA-seq can distinguish immune cell subtypes in a blood sample, like T-cells, B-cells, and macrophages.
        - **Pros**: High resolution, powerful for complex tissues.
        - **Cons**: High cost and computational complexity.

    2. **Bulk RNA-seq**: Measures an _aggregate_ gene expression across _all cells_ in a sample, giving a more generalized view.
        - **Example**: Analyzing tumor samples to detect overall differences in gene expression between cancerous and normal tissue.
        - **Pros**: Cost-effective and simpler.
        - **Cons**: Lacks cell type specificity.

-   **Deconvolution**: A computational technique used to estimate the proportions of different cell types from bulk RNA-seq data, effectively "reverse engineering" it.

    -   **Example**: Using deconvolution, you can estimate the percentage of immune cells in a blood sample versus non-immune cells.
    -   **Real-World Use Case**: Identifying changes in immune cell composition in diseases like cancer, COVID-19, or COPD.

## **Background**

### $S$: Single-Cell Matrix

-   **Dimensions:** $c$ cells $×$ $p$ genes ($c ≈ 5000$, $p ≈ 200$) for one patient.
    -   Multiple patients **stacked vertically**; real dimensions: ($n$ patients × $c$ cells) $×$ $p$ genes.
-   Each entry is a GEP for an **individual cell**.
-   **Expensive** to generate (requires cellular-level sampling).

### $B$: Bulk Matrix

-   **Dimensions:** $n$ patients $×$ $p$ genes.
-   Each entry is a GEP for an **entire** tissue (no cell-specific information).
-   Cheap to generate (high-level/abstract view).

### $R$: Reference Matrix

-   **Dimensions:** $k$ cell types $×$ $p$ genes ($k$ < 20).
-   **Aggregation** (average) of cell GEPs by cell type from **$S$**.

### $C$: cell type Abundance Matrix

-   **Dimensions:** $k$ cell types $×$ $n$ patients.
-   Each entry represents the **abundance** (fraction) of each cell type for a given GEP vector or matrix.
-   $R × C = B'$ (**pseudo-bulk**; like **$B$** but derived from single-cell data instead of different bulk sequencing technology).

### Relationship: $B ≈ C × R$

-   Since **$B$** and **$S$** samples are derived using _different technologies_, $B = C × R$ does **not** hold.
-   **Goal:** Learn a non-linear relationship from **$B$** to **$C$**.
-   **Outcome:** Estimated **$C$** can be multiplied with any reference GEP matrix **$R$** to produce a single-cell-adjusted bulk vector that has more accurate GEPs than the input bulk.

## **Previous Approach: Scaden**

Scaden Pipeline Visualization:

<div align="center">
    <img src="https://github.com/user-attachments/assets/cb51b76f-87b5-4f72-9424-dd8a96dd5c40" alt="Scaden" width="400"/>
</div>

### Methodology

-   Generates **synthetic** bulk samples by **randomly sampling cells** from the S matrix and **summing** them to get a vector. Repeats $n$ times to generate the pseudo-bulk matrix $B'$ ($p$ genes $×$ $n$ patients) by stacking individual synthetic bulk vectors.

    -   Each synthetically generated bulk vector will also have a corresponding **cell type abundance vector** which is generated by calculating the proportion (fraction) of each cell type in the set of randomly selected cells.

-   Trains a **deep learning** (DL) model that learns the relationship between synthetic bulk vectors and cell type abundance fractions.

    -   The model takes unseen bulk vectors $B_{new}$ as input and predicts cell type fractions $C$ as output.

### Datasets

-   **PBMC ~10k cells**: [Filtered single-cell matrix (HDF5)](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-10-k-1-standard-2-0-0)
-   **PBMC ~3k cells**: [Filtered single-cell matrix (HDF5)](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-granulocytes-removed-through-cell-sorting-3-k-1-standard-2-0-0)
-   **ACT Annotations**: [Web Tool](http://xteam.xbio.top/ACT/index.jsp); demo option generates cell type annotations for the PBMC 3k dataset.

## Novel Approach: Using Paired Data (2dRNA)

### Methodology

-   Utilizes **paired** bulk and single-cell RNA-seq data (from the same patients) to directly learn a non-linear relationship between bulk GEPs and cell type abundance fractions.
-   Trains a **deep learning** (DL) model that learns the relationship between **real, paired** bulk vectors and cell type abundance fractions.
    -   The model takes unseen bulk vectors $B_{new}$ as input and predicts cell type fractions $C$ as output.

### Improvements

-   Captures **realistic biological variation** across patients, improving generalizability.
-   Accounts for **technological difference** between bulk and single-cell sampling/sequencing.
-   Eliminates reliance on assumptions about random sampling during pseudo-bulk creation, resulting in a more **robust** model.

## **Related Works**

### **1. Scaden** (Menden et al., _Science Advances_ 2020)

-   **Approach:** Non-linear deep learning model to predict cell type proportions from bulk RNA-seq using single-cell data.
-   **Outcome:** More accurate deconvolution than linear models when trained on large, representative datasets.
-   **Methods:** Neural networks trained on synthetic bulk samples created from single-cell datasets.
-   **Evaluation:** Compares predicted cell type abundance fractions (**C'**) against ground-truth cell type proportions derived from paired bulk and single-cell RNA-seq datasets (e.g., PBMC datasets). Metrics include **Mean Absolute Error (MAE)** and **correlation** between predicted and true proportions.

### **2. SQUID** (Cobos et al., _Genome Biology_ 2023)

-   **Approach:** Benchmarks methods to deconvolve bulk RNA-seq into cell type contributions; introduces SQUID.
-   **Outcome:** Strengths, limitations, and applicability of methods across tissues/conditions, and evaluation of SQUID's performance.
-   **Methods:** Compares bulk to single-cell reference profiles, estimating the fraction of each cell type.
-   **Comparison:** SQUID is considered the current SOTA **linear** deconvolution algorithm. Our approach focuses on a **non-linear model** to learn more complex relationships.

### **3. DSSC** (Wang et al., _BMC Genomics_ 2024)

-   **Approach:** Deconvolution algorithm to infer both cell type proportions and GEPs from bulk samples.
-   **Outcome:** Simultaneously infers cell type proportions and GEPs for various datasets.
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

-   **Approach:** Predicts cell type proportions from bulk RNA-seq and adjusts bulk GEPs accordingly.
-   **Outcome:** Provides cell type proportions and adjusted bulk GEPs.
-   **Methods:** Linear deconvolution and reference profiles from single-cell data, using batch correction and smoothing.
-   **Comparison:** CIBERSORTx uses a **linear method** and signature gene sets; our approach learns a **non-linear relationship**.

### **7. Computational Deconvolution of Transcriptomics Data from Mixed Cell Populations** (Cobos et al., _Oxford Bioinformatics_ 2018)

-   **Approach:** Review of cell type deconvolution methods for bulk RNA-seq samples; good **starting point** for beginners to understand terminology.
-   **Outcome:** Provides a foundational understanding of cellular deconvolution and how single-cell data informs bulk deconvolution.

### **8. CIBERSORT** (Newman et al., _Nature Methods_ 2015)

-   **Approach:** Pioneering deconvolution method for estimating cell type proportions in bulk RNA-seq or microarray samples using a known set of "signature" genes for each cell type.
-   **Outcome:** Accurately estimates the relative abundances of immune cell types in complex tissues, often used as a baseline for later deconvolution models.
-   **Methods:** Linear support vector regression (SVR) trained on purified cell type expression profiles to create reference signatures. Bulk RNA-seq data is deconvolved relative to these signatures using SVR to infer cell type proportions.
-   **Evaluation:** Compares estimated cell type proportions to known ground-truth proportions from flow cytometry and other orthogonal methods. Performance is measured using **correlation coefficients** and **error rates** relative to true proportions in validation datasets.
-   **Comparison:** Unlike later models (like CIBERSORTx and Scaden), CIBERSORT relies on **linear regression** and pre-defined signature gene sets, whereas newer approaches (like Scaden) use **non-linear deep learning** to learn more flexible relationships.
