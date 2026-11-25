# =========================================================
# ĐÁNH GIÁ MÔ HÌNH CHÍNH THỨC TRÊN TẬP TEST
# Tái tạo đúng Train/Validation/Test (70/15/15, random_state=42)
# từ dữ liệu gốc, rồi chấm điểm model đã lưu (credit_default_hgb.joblib)
# trên đúng 4.500 khách hàng Test. Đồng thời huấn luyện lại 3 mô hình
# baseline (LR/RF/HGB) trên 19 và 28 đặc trưng để lập bảng so sánh.
# =========================================================

import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)  # feature_engineering.py, predict.py nam o thu muc goc

from feature_engineering import create_features

OUT_DIR = os.path.join(BASE_DIR, "official_outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
METRIC_DIR = os.path.join(OUT_DIR, "metrics")
for d in (OUT_DIR, FIG_DIR, METRIC_DIR):
    os.makedirs(d, exist_ok=True)


def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Da luu bieu do:", path)


def ks_stat(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return (tpr - fpr).max()


# ---------------------------------------------------------
# 1. DOC DU LIEU GOC VA TAI TAO DUNG SPLIT CHINH THUC
# ---------------------------------------------------------
raw_path = os.path.join(ROOT_DIR, "data", "UCI_Credit_Card.csv")
df = pd.read_csv(raw_path)

df = df.rename(columns={
    "default.payment.next.month": "default payment next month",
})
target_col = "default payment next month"

X_all = df.drop(columns=[target_col, "ID"])
y_all = df[target_col]

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.30, stratify=y_all, random_state=42,
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42,
)

print("===== TAI TAO SPLIT CHINH THUC (70/15/15) =====")
print("Train:", X_train.shape, "Default rate:", round(y_train.mean() * 100, 2), "%")
print("Valid:", X_valid.shape, "Default rate:", round(y_valid.mean() * 100, 2), "%")
print("Test :", X_test.shape, "Default rate:", round(y_test.mean() * 100, 2), "%")
assert X_test.shape[0] == 4500, "So dong Test khong khop ky vong (4500) - kiem tra lai file CSV goc."

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING: 19 DAC TRUNG GOC VA 28 DAC TRUNG
# ---------------------------------------------------------
with open(os.path.join(ROOT_DIR, "credit_default_config.json"), encoding="utf-8") as f:
    model_config = json.load(f)

features_28 = model_config["feature_columns"]
features_19 = [
    "LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

X_train_19 = create_features(X_train, features_19)
X_test_19 = create_features(X_test, features_19)

X_train_28 = create_features(X_train, features_28)
X_test_28 = create_features(X_test, features_28)

print("\nX_train_19:", X_train_19.shape, "| X_test_19:", X_test_19.shape)
print("X_train_28:", X_train_28.shape, "| X_test_28:", X_test_28.shape)

# ---------------------------------------------------------
# 3. HUAN LUYEN LAI 3 MO HINH BASELINE TREN 19 VA 28 DAC TRUNG
# ---------------------------------------------------------
def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=42,
        ),
    }


feature_sets = {"19 dac trung goc": (X_train_19, X_test_19), "28 dac trung (co FE)": (X_train_28, X_test_28)}

ablation_rows = []
model_28_fitted = {}

print("\n===== HUAN LUYEN LAI BASELINE TREN TAP TEST =====")
for fs_name, (Xtr, Xte) in feature_sets.items():
    models = make_models()
    for model_name, model in models.items():
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = {
            "Bo dac trung": fs_name,
            "Mo hinh": model_name,
            "So dac trung": Xtr.shape[1],
            "AUC": roc_auc_score(y_test, proba),
            "KS": ks_stat(y_test, proba),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
        }
        ablation_rows.append(row)
        print(f"  {fs_name:<22} | {model_name:<24} AUC={row['AUC']:.4f} KS={row['KS']:.4f}")
        if fs_name == "28 dac trung (co FE)":
            model_28_fitted[model_name] = model

ablation_df = pd.DataFrame(ablation_rows).round(4)
ablation_df.to_csv(os.path.join(METRIC_DIR, "ablation_19_vs_28.csv"), index=False, encoding="utf-8-sig")

bang_3_mohinh = ablation_df[ablation_df["Bo dac trung"] == "28 dac trung (co FE)"] \
    .drop(columns=["Bo dac trung"]).sort_values("AUC", ascending=False)
bang_3_mohinh.to_csv(os.path.join(METRIC_DIR, "so_sanh_3_mohinh_test.csv"), index=False, encoding="utf-8-sig")

print("\n=== BANG SO SANH 3 MO HINH (28 dac trung, tren Test) ===")
print(bang_3_mohinh.to_string(index=False))

print("\n=== BANG ABLATION 19 vs 28 DAC TRUNG (tren Test) ===")
print(ablation_df.to_string(index=False))

# ---------------------------------------------------------
# 4. MO HINH CHINH THUC (credit_default_hgb.joblib) TREN TEST
# ---------------------------------------------------------
final_model = joblib.load(os.path.join(ROOT_DIR, model_config["model_file"]))
final_threshold = float(model_config["decision_threshold"])

test_proba = final_model.predict_proba(X_test_28)[:, 1]
test_pred = (test_proba >= final_threshold).astype(int)

final_metrics = {
    "Model": model_config["model_name"],
    "Threshold": final_threshold,
    "AUC": roc_auc_score(y_test, test_proba),
    "KS": ks_stat(y_test, test_proba),
    "Accuracy": accuracy_score(y_test, test_pred),
    "Precision": precision_score(y_test, test_pred, zero_division=0),
    "Recall": recall_score(y_test, test_pred, zero_division=0),
    "F1": f1_score(y_test, test_pred, zero_division=0),
}
pd.DataFrame([final_metrics]).round(4).to_csv(
    os.path.join(METRIC_DIR, "final_model_test_metrics.csv"), index=False, encoding="utf-8-sig"
)

print("\n===== MO HINH CHINH THUC TREN TEST (credit_default_hgb.joblib) =====")
for k, v in final_metrics.items():
    print(f"  {k}: {v}")

tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
print(f"\nConfusion Matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")

# ---------------------------------------------------------
# 5. ROC CURVE (mo hinh chinh thuc + 3 baseline de doi chieu)
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
fpr, tpr, _ = roc_curve(y_test, test_proba)
plt.plot(fpr, tpr, linewidth=2.4,
         label=f"HGB Tuned (chinh thuc), AUC={final_metrics['AUC']:.3f}")

for name, model in model_28_fitted.items():
    proba = model.predict_proba(X_test_28)[:, 1]
    fpr_b, tpr_b, _ = roc_curve(y_test, proba)
    auc_b = roc_auc_score(y_test, proba)
    plt.plot(fpr_b, tpr_b, linestyle="--", alpha=0.8, label=f"{name} (baseline), AUC={auc_b:.3f}")

plt.plot([0, 1], [0, 1], "k:", label="Doan bua")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Duong ROC tren tap Test (4.500 khach hang)")
plt.legend()
save_fig("01_roc_curve_test.png")

# ---------------------------------------------------------
# 6. FEATURE IMPORTANCE (permutation importance, model chinh thuc)
# ---------------------------------------------------------
perm = permutation_importance(
    final_model, X_test_28, y_test, scoring="roc_auc",
    n_repeats=5, random_state=42, n_jobs=-1,
)
imp = pd.Series(perm.importances_mean, index=X_test_28.columns).sort_values()
imp.to_csv(os.path.join(METRIC_DIR, "feature_importance_test.csv"), header=["AUC_drop"], encoding="utf-8-sig")

plt.figure(figsize=(8, 6))
imp.tail(12).plot(kind="barh", color="#1565c0")
plt.title("Yeu to anh huong nhat (Permutation Importance tren Test)")
plt.xlabel("Muc giam AUC khi xao tron yeu to nay")
save_fig("02_feature_importance_test.png")

# ---------------------------------------------------------
# 7. BIEU DO EDA (tren toan bo du lieu goc, khong remap/dedup)
# ---------------------------------------------------------
df["DEFAULT"] = df[target_col]

df["DEFAULT"].value_counts().sort_index().plot(kind="bar", color=["#2e7d32", "#c62828"])
plt.xticks([0, 1], ["Khong vo no (0)", "Vo no (1)"], rotation=0)
plt.title("Phan bo khach hang theo tinh trang vo no (30.000 khach)")
plt.ylabel("So khach hang")
save_fig("03_eda_default_distribution.png")

df.groupby("PAY_0")["DEFAULT"].mean().plot(kind="bar", color="#1565c0")
plt.title("Ty le vo no theo trang thai tra no gan nhat (PAY_0)")
plt.ylabel("Ty le vo no")
save_fig("04_eda_default_by_pay0.png")

age_grp = pd.cut(df["AGE"], bins=[20, 30, 40, 50, 60, 80])
df.groupby(age_grp, observed=True)["DEFAULT"].mean().plot(kind="bar", color="#6a1b9a")
plt.title("Ty le vo no theo nhom tuoi")
plt.ylabel("Ty le vo no")
save_fig("05_eda_default_by_age.png")

limit_grp = pd.qcut(df["LIMIT_BAL"], q=5)
df.groupby(limit_grp, observed=True)["DEFAULT"].mean().plot(kind="bar", color="#ef6c00")
plt.title("Ty le vo no theo nhom han muc the")
plt.ylabel("Ty le vo no")
save_fig("06_eda_default_by_limit.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ["SEX", "EDUCATION", "MARRIAGE"]):
    table = df.groupby(col)["DEFAULT"].mean() * 100
    ax.bar(table.index.astype(str), table.values, color="#546e7a")
    ax.set_title(f"Default Rate by {col}")
    ax.set_ylabel("Default Rate (%)")
save_fig("07_eda_default_by_categorical.png")

# ---------------------------------------------------------
# 8. TEST_PREDICTIONS.CSV (mo hinh chinh thuc, 4.500 khach)
# ---------------------------------------------------------
predictions_df = X_test.copy()
predictions_df["ACTUAL_DEFAULT"] = y_test.values
predictions_df["PREDICTED_PROBABILITY"] = test_proba
predictions_df["PREDICTED_DEFAULT"] = test_pred
predictions_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=True, index_label="source_index", encoding="utf-8-sig")
print("\nDa luu test_predictions.csv:", predictions_df.shape)

# ---------------------------------------------------------
# 9. VIEC 2: KIEM CHUNG 3 NHOM MAU GREEN / YELLOW / RED
# ---------------------------------------------------------
def risk_group(p):
    if p < 0.15:
        return "Green"
    elif p < 0.25:
        return "Yellow"
    else:
        return "Red"


predictions_df["RISK_GROUP"] = predictions_df["PREDICTED_PROBABILITY"].apply(risk_group)

risk_summary = predictions_df.groupby("RISK_GROUP").agg(
    So_khach_hang=("ACTUAL_DEFAULT", "size"),
    So_vo_no_thuc_te=("ACTUAL_DEFAULT", "sum"),
    Ty_le_vo_no_thuc_te=("ACTUAL_DEFAULT", "mean"),
).reindex(["Green", "Yellow", "Red"])
risk_summary["Ty_le_vo_no_thuc_te"] = (risk_summary["Ty_le_vo_no_thuc_te"] * 100).round(2)

risk_summary.to_csv(os.path.join(METRIC_DIR, "risk_group_validation.csv"), encoding="utf-8-sig")

print("\n===== VIEC 2: KIEM CHUNG NHOM MAU (tren 4.500 khach Test) =====")
print(risk_summary.to_string())

print("\nHOAN TAT CA HAI VIEC.")
