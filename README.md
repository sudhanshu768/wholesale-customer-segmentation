# Wholesale Customer Segmentation using K-Means

##  Project Overview
This project implements a K-Means clustering model to segment wholesale customers
based on their purchasing behavior. The goal is to identify meaningful customer
groups and deploy the model through an interactive Streamlit web application.

---

##  Dataset
**Source:** Kaggle – Wholesale Customers Dataset

**Features used:**
- Fresh
- Milk
- Grocery
- Frozen
- Detergents_Paper
- Delicassen

Non-behavioral features such as `Channel` and `Region` were removed.

---

##  Methodology

### 1️ Data Preprocessing
- Removed categorical columns
- Applied feature scaling using `StandardScaler`

### 2️ Model Training
- Used K-Means clustering
- Optimal number of clusters determined using:
  - Elbow Method
  - Silhouette Score
- Final number of clusters selected: **K = 4**

### 3️ Evaluation & Visualization
- PCA used for 2D visualization
- Cluster means analyzed for business interpretation

### 4️ Robustness Testing
- Removed 20% of data and re-evaluated clustering
- Minimal change in silhouette score confirmed model stability

---

## Streamlit Application
The trained model and scaler were saved using `joblib` and deployed via Streamlit.

**Features of the app:**
- Accepts user purchase inputs
- Applies the same preprocessing pipeline
- Predicts customer segment
- Displays business interpretation of the cluster

🔗 **Live App:** *https://wholesale-customer-segmentation-hax2tqycovwjtanfqhw8ru.streamlit.app/*

---

## 📁 Project Structure
