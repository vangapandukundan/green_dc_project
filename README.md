# 🌿 Green Computing Data Center
### Predictive Energy Optimization using Machine Learning

---

## 📌 About
A machine learning project that predicts energy consumption of a university data center, identifies wasteful "zombie servers", and provides an interactive dashboard to simulate CO₂ savings under different conditions.

---

## 📁 Project Structure
```
green_dc_project/
├── data/
│   ├── green_quantum_data_centers_2.csv   ← raw dataset
│   └── processed_data.csv                 ← generated after data_prep.py
├── notebooks/
│   └── exploration.ipynb                  ← EDA and correlation charts
├── src/
│   ├── data_prep.py                       ← data cleaning + feature engineering
│   ├── model.py                           ← Random Forest model training
│   └── app.py                             ← Streamlit dashboard
├── model.pkl                              ← saved trained model
├── feature_importance.png                 ← generated chart
└── requirements.txt
```

---

## ⚙️ Installation

```bash
C:\Python314\python.exe -m pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn
```

---

## 🚀 How to Run

Run these commands in order inside VS Code terminal:

**1. Data Preparation**
```bash
C:\Python314\python.exe src/data_prep.py
```

**2. Train Model**
```bash
C:\Python314\python.exe src/model.py
```

**3. Launch Dashboard**
```bash
C:\Python314\python.exe -m streamlit run src/app.py
```

Open browser at → **http://localhost:8501**

---

## 📊 Dashboard Features

| Feature | Description |
|---|---|
| ⚡ What-If Predictor | Sliders to simulate workload and predict energy |
| 🌍 City Comparison | CO₂ savings by moving to a cooler city |
| 🧟 Zombie Servers | Lists high-energy, low-compute wasteful servers |
| 📊 Workload Chart | Average energy per workload type |
| 🌿 CO₂ Distribution | Histogram of emissions across all servers |

---

## 🛠️ Tech Stack
`Python 3.14` `Pandas` `NumPy` `Scikit-learn` `Streamlit` `Matplotlib` `Seaborn`

---

## 👨‍💻 Author
V. Kundan — Green Computing College Assignment Project
