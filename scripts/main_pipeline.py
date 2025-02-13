"""
Multiomics Data Integration & Analysis Pipeline
-------------------------------------------------

This script demonstrates a full workflow for integrating and analyzing 
metagenomics (species counts) and metabolomics data, along with metadata,
including exploratory data analysis (EDA), dimensionality reduction (PCA, t-SNE),
and disease prediction using Random Forest classifiers. 

Detailed annotations and comments are included to help you understand each step.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, ConfusionMatrixDisplay

# ---------------------------
# Helper Function to Save and Show Plots
# ---------------------------
def save_and_show(filename):
    """
    Save the current plot to a file with high quality and then display it.
    
    Parameters:
        filename (str): The name (and optionally path) of the file to save the plot.
    """
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
# Load datasets
species_file = "species.counts.tsv"
metadata_file = "metadata.tsv"
metabolomics_file = "mtb.tsv"

species_df = pd.read_csv(species_file, sep="\t")
metadata_df = pd.read_csv(metadata_file, sep="\t")
metabolomics_df = pd.read_csv(metabolomics_file, sep="\t")

# Rename the first column to "Sample" for merging purposes
species_df.rename(columns={species_df.columns[0]: "Sample"}, inplace=True)
metabolomics_df.rename(columns={metabolomics_df.columns[0]: "Sample"}, inplace=True)

# Merge datasets on the "Sample" column to combine omics and metadata information
merged_df = metadata_df.merge(species_df, on="Sample").merge(metabolomics_df, on="Sample")

# Select the top 50 most variable features from each omics dataset for demonstration purposes
top_species = species_df.iloc[:, 1:].var().nlargest(50).index  # select species columns
top_metabolites = metabolomics_df.iloc[:, 1:].var().nlargest(50).index  # select metabolite columns

species_cols = [col for col in species_df.columns if col in top_species]
metabolomics_cols = [col for col in metabolomics_df.columns if col in top_metabolites]

# Log-transform the data to reduce skewness (using log1p to handle zeros)
merged_df[species_cols] = np.log1p(merged_df[species_cols])
merged_df[metabolomics_cols] = np.log1p(merged_df[metabolomics_cols])

# Normalize the features using standard scaling (mean=0, std=1)
scaler = StandardScaler()
merged_df[species_cols] = scaler.fit_transform(merged_df[species_cols])
merged_df[metabolomics_cols] = scaler.fit_transform(merged_df[metabolomics_cols])

# For visualizations: Extract the species name from the taxonomy string (assuming species name follows "s__")
def extract_species_name(taxonomy):
    if "s__" in taxonomy:
        return taxonomy.split("s__")[-1]
    else:
        return taxonomy

short_species_names = [extract_species_name(s) for s in species_cols]

# ---------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------
# a) Metadata Overview: Plot the distribution of Study Groups
plt.figure(figsize=(8, 4))
sns.countplot(x="Study.Group", data=merged_df, order=merged_df["Study.Group"].value_counts().index)
plt.title("Distribution of Study Groups")
plt.xlabel("Study Group")
plt.ylabel("Count")
plt.tight_layout()
save_and_show("fig_distribution_study_groups.png")

# b) Age Distribution Histogram
plt.figure(figsize=(8, 4))
sns.histplot(merged_df["Age"], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age (Years)")
plt.tight_layout()
save_and_show("fig_age_distribution.png")

# c) Fecal Calprotectin Distribution Histogram
plt.figure(figsize=(8, 4))
sns.histplot(merged_df["Fecal.Calprotectin"].dropna(), kde=True)
plt.title("Fecal Calprotectin Distribution")
plt.xlabel("Fecal Calprotectin")
plt.tight_layout()
save_and_show("fig_fecal_calprotectin_distribution.png")

# d) Spearman Correlation Heatmap between Top Species and Metabolites
# Create an empty DataFrame to store the correlations
corr_matrix = pd.DataFrame(index=short_species_names, columns=top_metabolites)
for sp, sp_col in zip(short_species_names, species_cols):
    for met in top_metabolites:
        corr, _ = spearmanr(merged_df[sp_col], merged_df[met])
        corr_matrix.loc[sp, met] = corr
corr_matrix = corr_matrix.astype(float)

plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Spearman Correlation Heatmap: Species vs Metabolites")
plt.xlabel("Metabolites")
plt.ylabel("Species")
plt.tight_layout()
save_and_show("fig_spearman_correlation_heatmap.png")

# e) PCA on Each Omics Data
def run_pca(data, n_components=5):
    """
    Perform PCA on the provided data.
    
    Parameters:
        data (DataFrame): Data for PCA.
        n_components (int): Number of principal components to compute.
    
    Returns:
        components (ndarray): Transformed principal components.
        pca (PCA object): Fitted PCA model.
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)
    return components, pca

# PCA for species data
species_pcs, pca_species = run_pca(merged_df[species_cols])
# PCA for metabolomics data
metabo_pcs, pca_metabo = run_pca(merged_df[metabolomics_cols])

# Plot PCA for species data (first two principal components)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=species_pcs[:, 0], y=species_pcs[:, 1], hue=merged_df["Study.Group"], palette="deep")
plt.title("PCA of Metagenomics (Species) Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
save_and_show("fig_pca_species.png")

# Plot PCA for metabolomics data (first two principal components)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=metabo_pcs[:, 0], y=metabo_pcs[:, 1], hue=merged_df["Study.Group"], palette="deep")
plt.title("PCA of Metabolomics Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
save_and_show("fig_pca_metabolomics.png")

# ---------------------------
# 3. Intermediate Integration of Omics Data
# ---------------------------
# Use latent features from PCA for intermediate integration.
n_latent = 10
species_latent, _ = run_pca(merged_df[species_cols], n_components=n_latent)
metabo_latent, _ = run_pca(merged_df[metabolomics_cols], n_components=n_latent)

# Concatenate the latent features from both omics datasets to form the integrated feature set
integrated_features = np.concatenate([species_latent, metabo_latent], axis=1)
integrated_df = pd.DataFrame(integrated_features, 
                             columns=[f"Spec_PC{i+1}" for i in range(n_latent)] + 
                                     [f"Metabo_PC{i+1}" for i in range(n_latent)])

# Append metadata (e.g., Study.Group)
integrated_df["Study.Group"] = merged_df["Study.Group"].values

# Create a binary target variable: IBD (includes CD and UC) vs Healthy (Control)
integrated_df["IBD_Status"] = integrated_df["Study.Group"].apply(lambda x: "IBD" if x in ["CD", "UC"] else "Healthy")

# ---------------------------
# 4. Disease Prediction using Integrated Omics Data: Random Forest Classification
# ---------------------------
def plot_confusion_matrix(y_true, y_pred, title):
    """
    Generate and plot a confusion matrix.
    
    Parameters:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        title (str): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    save_and_show(f"fig_confusion_matrix_{title.replace(' ', '_')}.png")

# Prepare data for classification tasks
X = integrated_df.drop(columns=["Study.Group", "IBD_Status"])
y_binary = integrated_df["IBD_Status"]
y_multi = integrated_df["Study.Group"]

# Split data into training (70%) and validation (30%) sets with stratification
X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
X_train_multi, X_val_multi, y_train_multi, y_val_multi = train_test_split(X, y_multi, test_size=0.3, random_state=42, stratify=y_multi)

def train_evaluate_rf(X_train, y_train, X_test, y_test, problem_type="binary"):
    """
    Train and evaluate a Random Forest classifier.
    
    Parameters:
        X_train, X_test (DataFrame): Training and testing features.
        y_train, y_test (Series): Corresponding target labels.
        problem_type (str): "binary" or "multiclass" to specify the type of classification.
    
    Returns:
        rf (RandomForestClassifier): Trained Random Forest model.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Print detailed classification metrics
    print(classification_report(y_test, y_pred))
    
    # Plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix ({problem_type} classification)")
    
    # Generate ROC Curve for evaluation
    if problem_type == "binary":
        y_prob = rf.predict_proba(X_test)[:, 1]
        # Convert labels to binary (IBD:1, Healthy:0)
        fpr, tpr, _ = roc_curve([1 if label=="IBD" else 0 for label in y_test], y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({problem_type} classification)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        save_and_show("fig_roc_curve_binary.png")
    else:
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        y_prob = rf.predict_proba(X_test)
        plt.figure(figsize=(8, 6))
        for i, class_label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'One-vs-Rest ROC Curve ({problem_type} classification)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        save_and_show("fig_roc_curve_multiclass.png")
    
    return rf

# Binary Classification: IBD vs Healthy
print("---- Binary Classification (IBD vs Healthy) using Integrated Omics ----")
rf_bin = train_evaluate_rf(X_train_bin, y_train_bin, X_val_bin, y_val_bin, problem_type="binary")

# Multiclass Classification: CD vs UC vs Control
print("\n---- Multiclass Classification (CD vs UC vs Control) using Integrated Omics ----")
rf_multi = train_evaluate_rf(X_train_multi, y_train_multi, X_val_multi, y_val_multi, problem_type="multiclass")

# Feature Importance Plot from the binary Random Forest model
importances = rf_bin.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(10, 4))
sns.barplot(x=feat_imp.index, y=feat_imp.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Feature Importances (Integrated Latent Features) - Binary RF")
plt.tight_layout()
save_and_show("fig_feature_importances.png")

# PCA on the integrated data for overall clustering visualization
pca_integrated = PCA(n_components=2)
integrated_pcs = pca_integrated.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=integrated_pcs[:, 0], y=integrated_pcs[:, 1], hue=y_multi, palette="deep")
plt.title("PCA on Integrated Multiomics Data")
plt.xlabel("Integrated PC1")
plt.ylabel("Integrated PC2")
plt.tight_layout()
save_and_show("fig_pca_integrated.png")

# ---------------------------
# 5. Additional Analyses & Visualizations (Omics Data)
# ---------------------------
# a) t-SNE Visualization on Integrated Data
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y_multi, palette="deep")
plt.title("t-SNE on Integrated Multiomics Data")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
save_and_show("fig_tsne_integrated.png")

# b) Violin Plots for Selected Integrated Latent Features by Study Group
top_latent_features = X.columns[:4]  # display the first 4 latent features
for feat in top_latent_features:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x=y_multi, y=X[feat], palette="pastel")
    plt.title(f"Distribution of {feat} by Study Group")
    plt.xlabel("Study Group")
    plt.ylabel(feat)
    plt.tight_layout()
    save_and_show(f"fig_violin_{feat}.png")

# c) Correlation of Integrated Latent Features with Age
age_corr = {}
for col in X.columns:
    corr, _ = spearmanr(X[col], merged_df["Age"])
    age_corr[col] = corr
age_corr_df = pd.Series(age_corr).sort_values()

plt.figure(figsize=(8, 4))
sns.barplot(x=age_corr_df.index, y=age_corr_df.values, palette="coolwarm")
plt.title("Spearman Correlation of Integrated Features with Age")
plt.xlabel("Integrated Feature")
plt.ylabel("Correlation with Age")
plt.xticks(rotation=45)
plt.tight_layout()
save_and_show("fig_age_correlation_integrated_features.png")

# ---------------------------
# 6. Metadata Analysis & Random Forest Classification using Metadata Features
# ---------------------------
# Extract relevant metadata features including clinical variables and medication usage
metadata_features = merged_df[["Study.Group", "Age", "Fecal.Calprotectin", 
                               "antibiotic", "immunosuppressant", "mesalamine", "steroids"]].copy()

# Fill missing medication values with "No" (assuming missing implies no medication usage)
for med in ["antibiotic", "immunosuppressant", "mesalamine", "steroids"]:
    metadata_features[med] = metadata_features[med].fillna("No")

# Plot medication usage by Study Group for each medication type
for med in ["antibiotic", "immunosuppressant", "mesalamine", "steroids"]:
    plt.figure(figsize=(6,4))
    sns.countplot(x=med, hue="Study.Group", data=metadata_features, palette="Set2")
    plt.title(f"{med.capitalize()} Usage by Study Group")
    plt.tight_layout()
    save_and_show(f"fig_medication_usage_{med}.png")

# Box plot for Age by Study Group
plt.figure(figsize=(6,4))
sns.boxplot(x="Study.Group", y="Age", data=metadata_features, palette="Set3")
plt.title("Age by Study Group")
plt.tight_layout()
save_and_show("fig_boxplot_age_by_study_group.png")

# Box plot for Fecal Calprotectin by Study Group
plt.figure(figsize=(6,4))
sns.boxplot(x="Study.Group", y="Fecal.Calprotectin", data=metadata_features, palette="Set3")
plt.title("Fecal Calprotectin by Study Group")
plt.tight_layout()
save_and_show("fig_boxplot_calprotectin_by_study_group.png")

# Create a new binary target variable for metadata-based classification: IBD vs Healthy
metadata_features["IBD_Status"] = metadata_features["Study.Group"].apply(lambda x: "IBD" if x in ["CD", "UC"] else "Healthy")

# Convert categorical medication columns into dummy/indicator variables
medication_dummies = pd.get_dummies(metadata_features[["antibiotic", "immunosuppressant", "mesalamine", "steroids"]], drop_first=True)

# Combine numeric metadata features with the medication dummy variables
X_meta = pd.concat([metadata_features[["Age", "Fecal.Calprotectin"]], medication_dummies], axis=1)
y_meta_binary = metadata_features["IBD_Status"]
y_meta_multi = metadata_features["Study.Group"]

# Split the metadata features into training and validation sets (70/30 stratified)
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(X_meta, y_meta_binary, test_size=0.3, random_state=42, stratify=y_meta_binary)

def train_evaluate_rf_metadata(X_train, y_train, X_test, y_test, problem_type="binary"):
    """
    Train and evaluate a Random Forest classifier using metadata features.
    
    Parameters:
        X_train, X_test (DataFrame): Training and testing features.
        y_train, y_test (Series): Corresponding target labels.
        problem_type (str): "binary" for IBD vs Healthy.
    
    Returns:
        rf (RandomForestClassifier): Trained Random Forest model.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    print("Metadata RF Classification Report:")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, f"Metadata Confusion Matrix ({problem_type})")
    
    # ROC Curve for binary classification
    if problem_type == "binary":
        y_prob = rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve([1 if label=="IBD" else 0 for label in y_test], y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Metadata ROC Curve (Binary)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        save_and_show("fig_metadata_roc_curve.png")
    
    return rf

print("\n---- Metadata-based RF Classification (IBD vs Healthy) ----")
rf_meta = train_evaluate_rf_metadata(X_meta_train, y_meta_train, X_meta_val, y_meta_val, problem_type="binary")
