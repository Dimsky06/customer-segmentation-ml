# 📊 Customer Segmentation using Unsupervised Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This repository contains an end-to-end **Unsupervised Machine Learning** pipeline designed for **Customer Segmentation**. By analyzing demographic and transactional data, this project applies data preprocessing, Dimensionality Reduction via **PCA (Principal Component Analysis)**, and **K-Means Clustering** to segment customers into actionable behavioral groups.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `LabelEncoder`, `PCA`, `KMeans`)
* **Visualization:** `matplotlib`, `seaborn`, `yellowbrick`
* **Model Export:** `joblib`, `h5py`

---

## ⚡ Machine Learning Pipeline
1. **Data Cleaning & Preprocessing:** Handling missing values, categorical encoding (`LabelEncoder`), and feature scaling (`StandardScaler`).
2. **Dimensionality Reduction (PCA):** Reducing feature dimensions while retaining maximum variance for 2D visualization.
3. **Model Building & Clustering Evaluation:** Determining optimal cluster count ($k$) using **Elbow Method** and **Silhouette Score**.
4. **Cluster Interpretation:** Profiling cluster centroids to extract business insights.
5. **Model Export:** Saving the trained K-Means model (`model_clustering.h5`) and PCA transformation model (`PCA_model_clustering.h5`).

---

## 📁 Repository Structure
```text
.
├── [Clustering]_Submission_Akhir_BMLP_Dimas_Eka_Maulana.ipynb  # Primary Jupyter Notebook
├── data_clustering.csv                                        # Processed Dataset
├── data_clustering_inverse.csv                                # Inverse Scaled Dataset
├── model_clustering.h5                                        # Trained K-Means Model
├── PCA_model_clustering.h5                                    # Trained PCA Model
└── README.md                                                  # Project Documentation
