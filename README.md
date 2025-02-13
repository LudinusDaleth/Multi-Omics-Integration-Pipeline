# Multi-Omics Integration & Analysis Pipeline

## Description
This repository hosts a Python-based pipeline that integrates metagenomic (species counts) and metabolomic data alongside clinical metadata. The workflow demonstrates data preprocessing, dimension reduction (PCA, t-SNE), correlation analysis, and disease classification using Random Forest. It’s designed to handle high-dimensional multi-omics datasets and provide an end-to-end analytic framework from raw data to predictive modeling.

---

## Key Features
- **Data Merging & Preprocessing**: Automatic alignment of samples across omics and metadata, log-transformation, and standard scaling.
- **Feature Selection**: Isolation of highly variable species and metabolites for efficient dimensionality reduction.
- **Exploratory Analysis**: Histograms, correlation heatmaps, PCA, and t-SNE visualizations to uncover data patterns.
- **Intermediate Integration**: Latent features (principal components) from both omics layers are combined into a single integrated feature matrix.
- **Classification**: Binary (IBD vs. Healthy) and multiclass (CD, UC, Control) Random Forest models to evaluate predictive power.
- **Metadata-Driven**: Additional analyses focusing on clinical features (e.g., Age, Fecal Calprotectin, medication usage).

---

## Requirements
- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- SciPy  

You can install all dependencies using:  
```bash
pip install -r requirements.txt
