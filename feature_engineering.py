
import numpy as np
import pandas as pd

STATUS_COLS = [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

ORIGINAL_FEATURES = (
    ["LIMIT_BAL"]
    + STATUS_COLS
    + BILL_COLS
    + PAYMENT_COLS
)


def create_features(data, feature_columns):
    """
    Chuyển 19 đặc trưng tài chính gốc
    thành 28 đặc trưng dùng cho mô hình AI.

    Parameters
    ----------
    data : pandas.DataFrame
        Dữ liệu khách hàng với 19 cột gốc.

    feature_columns : list
        Danh sách 28 đặc trưng từ model config.

    Returns
    -------
    pandas.DataFrame
        Dữ liệu đã Feature Engineering,
        đúng thứ tự cột mô hình yêu cầu.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input phải là pandas DataFrame.")

    missing = [
        col for col in ORIGINAL_FEATURES
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu các cột đầu vào: {missing}"
        )

    result = data[ORIGINAL_FEATURES].copy()

    # Kiểm tra dữ liệu số
    for col in ORIGINAL_FEATURES:
        result[col] = pd.to_numeric(
            result[col], errors="raise"
        )

    if result.isna().any().any():
        raise ValueError("Dữ liệu có giá trị thiếu.")

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Dữ liệu chứa giá trị không hợp lệ.")

    if (result["LIMIT_BAL"] <= 0).any():
        raise ValueError("LIMIT_BAL phải lớn hơn 0.")

    # 1. Nhóm lịch sử thanh toán
    result["DELAY_COUNT"] = (
        result[STATUS_COLS] > 0
    ).sum(axis=1)

    result["SEVERE_DELAY_COUNT"] = (
        result[STATUS_COLS] >= 2
    ).sum(axis=1)

    result["MAX_DELAY"] = (
        result[STATUS_COLS].max(axis=1)
    )

    # 2. Nhóm số tiền thanh toán
    result["ZERO_PAYMENT_COUNT"] = (
        result[PAYMENT_COLS] == 0
    ).sum(axis=1)

    result["TOTAL_PAYMENT_6M"] = (
        result[PAYMENT_COLS].sum(axis=1)
    )

    # 3. Nhóm số tiền sao kê
    result["AVG_BILL_6M"] = (
        result[BILL_COLS].mean(axis=1)
    )

    result["BILL_CHANGE"] = (
        result["BILL_AMT1"] - result["BILL_AMT6"]
    )

    # 4. Nhóm tỷ lệ sao kê / hạn mức
    result["UTILIZATION_RATIO"] = (
        result["BILL_AMT1"] / result["LIMIT_BAL"]
    )

    result["AVG_UTILIZATION_6M"] = (
        result["AVG_BILL_6M"] / result["LIMIT_BAL"]
    )

    # Đảm bảo đúng 28 cột và đúng thứ tự
    missing_model_features = [
        col for col in feature_columns
        if col not in result.columns
    ]

    if missing_model_features:
        raise ValueError(
            "Thiếu đặc trưng mô hình: "
            f"{missing_model_features}"
        )

    result = result[feature_columns].copy()

    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(
            "Feature Engineering tạo giá trị không hợp lệ."
        )

    return result
