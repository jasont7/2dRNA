# Cellular Deconvolution: A Deep Learning Approach to Cell Type Fraction Estimation

## 1. Introduction

The study of tissue-specific gene expression through next-generation sequencing, particularly RNA sequencing (RNA-seq), plays a crucial role in understanding biological and disease processes. However, a major challenge of bulk RNA-seq is that it captures an aggregate gene expression signal from a heterogeneous mixture of cell types, each with distinct functional states. Differences in gene expression between conditions may result from shifts in the cellular composition of the tissue, intrinsic changes within specific cell populations, or a combination of both. Accurately disentangling these factors is especially critical in diseases characterized by cell proliferation, such as cancer, or by progressive cell loss, as seen in neurodegenerative disorders and chronic conditions like chronic obstructive pulmonary disease (COPD), where alterations in tissue composition can confound gene expression analysis.

Cellular deconvolution is the process of estimating cell type fractions in a **tissue sample** from bulk RNA sequencing (RNA-seq) data. Bulk RNA-seq provides an aggregate measure of gene expression profiles (GEPs) across **all cells** in a tissue, making it impossible to distinguish GEP contributions from each individual cell type. This limitation has driven the development of computational methods aimed at deconvolving bulk RNA-seq GEPs into meaningful cell type proportions.

Existing approaches to cellular deconvolution fall into two main categories: linear and nonlinear methods. Linear methods, such as CIBERSORT (which uses Non-negative Least Squares, NNLS), assume that the gene expression measured in bulk RNA-seq is a **weighted sum** of reference gene expression signatures, where the weights correspond to the proportions of each cell type in the sample. Here, **reference gene expression signatures** refer to characteristic gene expression patterns associated with specific cell types, typically derived from single-cell RNA sequencing (scRNA-seq) or sorted cell populations.

More advanced linear methods incorporate additional constraints and prior knowledge to improve accuracy. For example, constrained regression approaches, such as CIBERSORTx, use statistical regularization techniques like ridge regression to reduce overfitting. Bayesian deconvolution models, such as BSEQ-sc, encode prior distributions over cell type proportions to improve robustness. Other methods, such as MuSiC, leverage **reference expression matrices**, which are collections of reference gene expression signatures constructed from scRNA-seq data, to refine predictions. Despite these improvements, linear models struggle to capture the **nonlinear relationships** present in gene expression data, particularly in complex tissues where cell-cell interactions and regulatory effects alter gene expression in ways that cannot be represented as a simple weighted sum.

Nonlinear methods, particularly deep learning approaches, aim to overcome these limitations by learning complex mappings between bulk RNA-seq and cell type proportions. Scaden (2020), is the most widely known nonlinear deconvolution method, which employs deep neural networks (DNNs) trained to infer cell type proportions directly from bulk RNA-seq. However, Scaden relies on training with **pseudo-bulk** RNA-seq data rather than **real paired** bulk and single-cell RNA-seq (scRNA-seq) data. Here, **pseudo-bulk** refers to bulk RNA-seq profiles that are computationally generated by aggregating expression values of individual cells, mimicking an experimentally measured bulk sample. In contrast, **real paired** data consists of experimentally measured bulk RNA-seq samples with corresponding scRNA-seq profiles from the **same biological source**, such as a patient tissue sample. The key difference is that pseudo-bulk data lacks technical noise and batch effects inherent in real experimental bulk measurements, potentially leading to models that generalize poorly to real-world data.

The **first research question** we address is: Can deep learning-based deconvolution models effectively learn from real-world paired bulk and single-cell RNA-seq data, which exhibit biological variability and technical noise, as opposed to performing well only on controlled or computationally generated datasets?

To explore this, we develop a deep learning model trained on real paired bulk-scRNA samples, allowing it to learn from the inherent complexities of experimental data. Unlike pseudo-bulk data, real paired datasets exhibit biological variability and technical noise introduced by differences in sequencing protocols, RNA extraction methods, and measurement discrepancies between bulk and single-cell technologies. These factors can lead to systematic biases, such as differences in gene capture efficiency and sequencing depth, which pseudo-bulk data does not replicate. By training on real paired data, our model is better suited for practical applications where such variations naturally occur in bulk RNA-seq samples.

The second research question we address is: What is the optimal neural network architecture for deconvolving real paired bulk-scRNA data, balancing model complexity, generalization ability, and computational efficiency?

To address this, we develop a novel deep neural network (DNN) architecture tailored to real paired datasets. Unlike Scaden, which uses a fixed ensemble architecture, we optimize the network structure through hyperparameter tuning of variables such as network depth, layer sizes, and dropout rates. We also enhance model efficiency by selecting only informative feature genes, reducing the size of the input bulk RNA-seq vector. This dimensionality reduction improves computational efficiency while preserving the most relevant information for accurate cell type proportion estimation. These enhancements result in greater accuracy, improved robustness across different tissue types, and better generalization to unseen datasets, making our approach more reliable for real-world deconvolution tasks.

In summary, our contributions are:

1. **Training on real paired bulk-scRNA data** – Unlike previous deep learning-based deconvolution models that rely on pseudo-bulk data, we train on experimentally measured real paired bulk and single-cell RNA-seq samples. This allows our model to learn from biological variability and technical noise inherent in real-world datasets, improving its applicability to practical scenarios.
2. **Developing an optimized neural network architecture** – We design a deep neural network (DNN) architecture specifically tailored for real paired bulk-scRNA deconvolution. Instead of using a fixed architecture like Scaden, we apply hyperparameter tuning to optimize network depth, layer sizes, and dropout rates for improved generalization and robustness.
3. **Improving model efficiency through feature selection** – To enhance computational efficiency, we reduce the input dimension by selecting only informative feature genes from the bulk RNA-seq vector. This ensures the model focuses on the most relevant signals while maintaining deconvolution accuracy and reducing unnecessary complexity.

## 2. Scaden: A Nonlinear Approach Using Simulated Data

Scaden is a deep learning model designed for cellular deconvolution that employs an ensemble of three feedforward neural networks to predict cell-type fractions. The model averages predictions from each of the three networks to improve robustness and reduce variance. Unlike traditional linear methods, Scaden is trained to capture complex, nonlinear relationships between gene expression and cell type proportions.

Scaden's training process involves generating synthetic bulk RNA-seq data from scRNA-seq samples. This synthetic data is created by aggregating expression profiles of randomly sampled single cells to simulate realistic bulk RNA-seq experiments. The advantage of this approach is that it allows Scaden to train on a large dataset that spans a wide range of possible gene expression profiles. However, the reliance on simulated data introduces limitations. The model may not generalize well to real bulk RNA-seq data, as the synthetic samples may not fully capture **real biological variability**. Furthermore, Scaden employs a fixed neural network architecture, which may not be optimal for all datasets, limiting its adaptability to new datasets such as ours and potentially affecting its accuracy.

### 2.1 Scaden Architecture in Depth

Scaden consists of an ensemble of three independent deep neural networks (DNNs), each trained separately and then averaged at inference time. Each network has four hidden layers with varying sizes, ranging from 32 to 1024 neurons per layer, and employs dropout for regularization. The final layer applies a softmax activation function to produce normalized cell-type fractions. Training is performed using the Adam optimizer with an initial learning rate of 0.0001, and early stopping is applied after 5000 iterations to prevent overfitting.

### 2.2 Scaden Training Dataset Simulation in Depth

Scaden's training dataset is generated from scRNA-seq data through a **bulk-simulation strategy** (i.e., bulk samples are generated from single-cell data). Single-cell profiles are randomly selected from a reference scRNA-seq dataset, and their expression values are aggregated to form a **pseudo-bulk sample**. The proportions of selected cells are assigned using a random uniform distribution, ensuring variability across training samples. This results in tens of thousands of synthetic bulk samples spanning multiple tissue types. However, since these samples do not originate from real patient bulk RNA-seq data, they may fail to capture technological effects present in real sequencing experiments.

## 3. Our Novel Paired Dataset and Data Augmentation Strategy

To address the shortcomings of Scaden, we introduce a novel dataset consisting of **paired bulk and scRNA-seq data** from approximately 30 patients, where **paired** means that each bulk sample corresponds to a matched scRNA-seq sample from the **same patient**. Unlike Scaden’s synthetic training data, our dataset captures **real biological variability**, providing a more reliable foundation for model training.

### 3.1 Augmentation Strategy

Given the limited size of our dataset (`n = ~30`), we employ a **random sampling augmentation strategy** to expand our training data. Specifically, we generate additional training samples by randomly sampling subsets of single-cell data from each patient’s scRNA-seq sample.

For each patient:

-   Repeat `n_aug` times:
    -   A fraction (`aug_ratio`) of available single-cell profiles is randomly sampled.
    -   The cell-type fractions are calculated based on the sampled subset.
-   The patient's corresponding bulk sample vector is paired with `n_aug` cell-type fraction vectors, rather than a one-to-one pairing, growing the size of the entire dataset by a factor of `n_aug`.

This augmentation technique introduces controlled variability while preserving patient-specific expression characteristics, improving generalization and robustness for model training.

## 4. Evaluating Scaden on Paired Data

Our primary goal is to establish a **new benchmark** for cellular deconvolution using real paired data. We retrain Scaden on our paired dataset and compare its performance to models trained on synthetic data. Rather than aiming for real paired data to improve model performance, we aim to establish a more **realistic evaluation setting**.

Our key findings indicate that Scaden, when trained on real paired data, performs comparably to its synthetic-data-trained version {{MUST CONFIRM THIS}} while significantly outperforming existing linear methods previously evaluated on the same dataset (as established in our prequel paper). This confirms that nonlinearity plays a larger role in performance improvements than data source alone, suggesting that Scaden’s main advantage over linear methods stems from its ability to capture complex relationships.

## 5. Our Proposed Method: SimpleDNN

To further improve upon Scaden, we introduce **SimpleDNN**, a deep learning model designed to optimize cellular deconvolution. SimpleDNN builds upon Scaden’s framework to achieve optimal performance given our new paired dataset.

One of the key improvements in SimpleDNN is the use of **hyperparameter tuning** to search for the optimal neural network architecture. Unlike Scaden, which relies on a fixed architecture, SimpleDNN explores different configurations, including varying hidden layer sizes, depths, dropout rates, and learning rates, to determine the best-performing model. This flexibility allows SimpleDNN to adapt to different datasets more effectively.

Additionally, SimpleDNN is implemented in **PyTorch**, addressing the maintainability issues associated with Scaden’s outdated TensorFlow implementation, which has not been updated in over four years. This transition improves performance, facilitates integration with modern deep learning tools, and enables fairer comparisons between different models.

### 5.1 Experimental Results

Our experimental results demonstrate that SimpleDNN significantly outperforms Scaden (both trained and evaluated on our paired dataset). We assess performance using multiple evaluation metrics, including mean squared error (MSE), R², KL divergence, and Wasserstein distance.

| Metric               | Scaden    | SimpleDNN | Improvement (%) |
| -------------------- | --------- | --------- | --------------- |
| MSE                  | 0.0001557 | 0.0000051 | 96.72%          |
| R²                   | 0.9971277 | 0.9999056 | 0.28%           |
| KL Divergence        | 0.0001256 | 0.0000037 | 97.05%          |
| Wasserstein Distance | 0.0060528 | 0.0010639 | 82.42%          |

These results indicate that hyperparameter tuning leads to consistent improvements across all evaluation metrics, further supporting the advantage of our method over Scaden.

## 6. Discussion

### 6.1 Deep Learning vs. Linear Methods

Our results confirm that deep learning models, such as Scaden and SimpleDNN, capture complex nonlinear relationships in gene expression data more effectively than traditional linear methods. However, deep learning approaches also require larger datasets, greater computational resources, and careful tuning. Our prequel paper introduces a novel sample collection protocol to inform biologists with the aim to address these considerations.

### 6.2 Migration from TensorFlow to PyTorch

Scaden’s TensorFlow implementation is outdated and difficult to maintain. By reimplementing it in **PyTorch**, we improve maintainability, performance, and compatibility with contemporary deep learning frameworks, making it easier to compare with SimpleDNN.

### 6.3 Limitations and Future Work

While our paired dataset provides a significant improvement over synthetic training data, it remains relatively small. Future work will focus on expanding the dataset and exploring semi-supervised learning techniques.

## References

(Todo)
