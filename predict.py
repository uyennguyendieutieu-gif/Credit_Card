
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from feature_engineering import (
    create_features,
    ORIGINAL_FEATURES
)

# Đường dẫn luôn tính từ vị trí file predict.py
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "credit_default_hgb.joblib"
CONFIG_PATH = BASE_DIR / "credit_default_config.json"

# Tải mô hình và cấu hình một lần
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

MODEL = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = CONFIG["feature_columns"]
DECISION_THRESHOLD = float(CONFIG["decision_threshold"])


def probability_to_score(probabilities):
    """
    Quy đổi xác suất default thành điểm nội bộ 300–850.
    Điểm càng cao thì rủi ro dự đoán càng thấp.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    return np.rint(
        300 + 550 * (1 - probabilities)
    ).astype(int)


def assign_risk_group(probability):
    """
    Phân nhóm theo xác suất default chưa làm tròn.
    """
    if probability < 0.15:
        return "Green"
    elif probability < DECISION_THRESHOLD:
        return "Yellow"
    return "Red"


def predict_batch(data):
    """
    Dự đoán nhiều khách hàng.

    Input:
        pandas.DataFrame chứa 19 đặc trưng tài chính gốc.

    Output:
        pandas.DataFrame gồm kết quả dự đoán
        và các chỉ số Credit Scoring.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "predict_batch yêu cầu pandas DataFrame."
        )

    if data.empty:
        raise ValueError("Dữ liệu đầu vào đang trống.")

    # Tạo đúng 28 đặc trưng
    X = create_features(data, FEATURE_COLUMNS)

    # Dự đoán xác suất default
    probabilities = MODEL.predict_proba(X)[:, 1]

    # Tính điểm và phân nhóm
    scores = probability_to_score(probabilities)

    groups = [
        assign_risk_group(float(p))
        for p in probabilities
    ]

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    # Giữ nguyên chỉ số hàng đầu vào
    results = pd.DataFrame({
        "Default_Probability": probabilities,
        "Credit_Score": scores,
        "Risk_Group": groups,
        "Predicted_Default": predictions
    }, index=data.index)

    return results


def predict_customer(customer):
    """
    Dự đoán một khách hàng.

    Input:
        dict gồm 19 đặc trưng tài chính gốc.

    Output:
        dict gồm xác suất, điểm, nhóm rủi ro
        và nhãn dự đoán.
    """

    if not isinstance(customer, dict):
        raise TypeError(
            "predict_customer yêu cầu dictionary."
        )

    customer_df = pd.DataFrame([customer])

    result = predict_batch(customer_df).iloc[0]

    return {
        "default_probability": float(
            result["Default_Probability"]
        ),
        "credit_score": int(
            result["Credit_Score"]
        ),
        "risk_group": str(
            result["Risk_Group"]
        ),
        "predicted_default": int(
            result["Predicted_Default"]
        )
    }
