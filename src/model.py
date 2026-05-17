import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score


def load_data():
    path = os.path.join("data", "processed_data.csv")
    df   = pd.read_csv(path)
    print(f"✅ Loaded processed data: {df.shape[0]} rows\n")
    return df


def prepare_features(df):
    # ── Only use real input features (no co2_kg — that is derived from target) ──
    features = [
        "compute_demand_TFlops",   # how much computing is needed
        "storage_demand_TB",       # how much storage is needed
        "network_demand_Gbps",     # how much network is needed
        "workload_encoded"         # type of workload (encoded as number)
    ]
    target = "energy_consumption_kWh"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train size: {len(X_train)} | Test size: {len(X_test)}\n")
    return X_train, X_test, y_train, y_test, features


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,    # more trees = better accuracy
        max_depth=10,        # prevent overfitting
        random_state=42
    )
    model.fit(X_train, y_train)
    print("🌲 Random Forest model trained!\n")
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"📈 Model Accuracy (R² Score) : {r2:.4f}  (closer to 1.0 = better)")
    print(f"📉 Mean Absolute Error        : {mae:,.2f} kWh\n")
    return preds


def plot_feature_importance(model, features):
    importance = model.feature_importances_
    idx        = np.argsort(importance)

    plt.figure(figsize=(8, 5))
    plt.barh([features[i] for i in idx], importance[idx], color="#2ecc71")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance – What drives energy consumption?")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("📊 Chart saved → feature_importance.png\n")


def save_model(model):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("💾 Model saved → model.pkl")


if __name__ == "__main__":
    df                                       = load_data()
    X_train, X_test, y_train, y_test, feats = prepare_features(df)
    model                                    = train_model(X_train, y_train)
    preds                                    = evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feats)
    save_model(model)
    print("\n✅ model.py completed successfully!")