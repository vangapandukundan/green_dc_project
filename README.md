# 🌿 Green Quantum Data Center Optimization

> An advanced Machine Learning and Interactive Dashboard platform designed to optimize data center energy consumption, minimize CO₂ emissions, and identify anomalous infrastructure behavior.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost%20%7C%20LightGBM-success.svg)
![Unsupervised ML](https://img.shields.io/badge/Unsupervised-IsolationForest%20%7C%20K--Means-blueviolet.svg)

## 📌 Project Overview
This project simulates, analyzes, and predicts the energy efficiency and carbon emissions of modern, high-performance datacenters. It acts as an **Active Decision Support System**, helping infrastructure engineers predict power consumption before deploying heavy compute workloads and automatically hunting down wasteful "zombie" servers.

It is split into two advanced Data Science pipelines:
1. **Supervised Learning Engine:** A highly tuned, cross-validated multi-model benchmark (RandomForest, XGBoost, LightGBM, GradientBoosting) to predict energy usage and CO₂ output based on compute demand, storage capacity, and security operations.
2. **Unsupervised Learning & Anomaly Detection:** An AI framework utilizing **Isolation Forests** to detect structural anomalies and **K-Means Clustering with 3D PCA dimension reduction** to categorize server behavioral topologies.

---

## ✨ Features & Architecture

### 📊 Comprehensive Interactive Dashboard
Built with **Streamlit** and heavily utilizing **Plotly** for rich interactive elements.
* **⚡ What-If Predictor:** Slide workload variables (Compute TFlops, Storage, Network) and instantly see predicted energy demands.
* **📊 Model Comparison Benchmark:** Automatically compares 4 regression algorithms using K-Fold CV metrics (R², MAE, RMSE).
* **🧠 SHAP Explainability:** Calculates Shapley Additive exPlanations to visually break down exactly *why* the model made specific predictions.
* **🌍 City CO₂ Analysis:** Demonstrates the geographic impact of data center locations, highlighting how local temperatures impact cooling coefficients and carbon footprints.
* **🧟 Zombie Server Detection:** Hunts down servers with high residual energy draw but negligible compute utilization.
* **🤖 Unsupervised ML Analysis:** Identifies anomalous servers and visualizes a 17-dimensional feature space mapped down to 3D via PCA K-Means clustering.

### 🧠 The Machine Learning Pipeline (`train.py`)
A single-click orchestration pipeline that executes the entire Data Science lifecycle:
1. Loads and validates raw configuration data.
2. Executes deep feature engineering, defining categorical mappings and efficiency indices.
3. Operates an **Optuna** Hyperparameter Optimization engine to find the theoretical limits of the decision trees.
4. Performs rigorous 5-Fold Cross Validation.
5. Saves the models, PCA matrices, and Shap explainer state natively to local serialized objects for instantaneous UI rendering.

---

## 🛠 Project Structure

```text
green_dc_project/
├── config.py              # Central configuration, parameters, and hyperparameters
├── train.py               # ML Training orchestration script
├── requirements.txt       # Python dependencies
├── src/
│   ├── data/
│   │   ├── loader.py      # Raw dataset ingestion
│   │   └── preprocessor.py# Feature engineering & encoding
│   ├── models/
│   │   ├── trainer.py     # Random Forest, XGBoost, LightGBM initialization
│   │   ├── evaluator.py   # R², MAE, and CV logic
│   │   ├── tuner.py       # Optuna Hyperparameter optimization
│   │   ├── explainer.py   # SHAP value generation
│   │   └── unsupervised.py# Isolation Forest & K-Means/PCA architecture
│   ├── dashboard/
│   │   ├── app.py         # Main Streamlit web application
│   │   └── components/    # Modular UI components (Tabs)
│   │       ├── predictor.py
│   │       ├── model_comparison.py
│   │       ├── explainability.py
│   │       ├── city_analysis.py
│   │       ├── zombie_analysis.py
│   │       └── unsupervised_analysis.py
```

---

## 🚀 Quickstart & Installation

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/green_dc_project.git
cd green_dc_project
```

**2. Install Dependencies:**
Ensure you have a modern Python 3 installation, then install required modules:
```bash
pip install -r requirements.txt
```

**3. Train the Core AI Models:**
You must generate the `model_results.pkl` artifact before booting the dashboard.
To train using default fast configurations:
```bash
python train.py
```
*(Optional)* To train using **Optuna** for deep hyperparameter optimization (Will take longer):
```bash
python train.py --tune
```

**4. Launch the Dashboard:**
```bash
streamlit run src/dashboard/app.py
```
*Navigate to `http://localhost:8501` in your browser.*

---

## 🔬 Technologies Used
* **Languages:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (RandomForest, GradientBoosting, IsolationForest, KMeans, PCA), XGBoost, LightGBM, Optuna
* **Interpretability:** SHAP
* **Frontend UI:** Streamlit, Plotly, HTML/CSS

---
