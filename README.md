# CreditAI - Machine Learning Model Handover

## 1. Project Overview

CreditAI là hệ thống hỗ trợ nhân viên ngân hàng
đánh giá rủi ro tín dụng dựa trên Machine Learning.

Mô hình dự đoán khả năng khách hàng không thực hiện
nghĩa vụ thanh toán trong kỳ tiếp theo (Default).

Model: Hist Gradient Boosting Tuned

Input: 19 đặc trưng tài chính gốc

Feature Engineering: Tự động tạo 9 đặc trưng

Model Input: 28 đặc trưng

Decision Threshold: 0.25

Output:
- Default Probability
- Credit Score (300-850)
- Risk Group (Green / Yellow / Red)
- Predicted Default (0 / 1)

Lưu ý: Credit Score là điểm nội bộ do dự án
tự thiết kế, không phải điểm tín dụng chính thức.

---

## 2. Project Files

credit_default_hgb.joblib
    Mô hình AI đã huấn luyện.

credit_default_config.json
    Cấu hình mô hình và danh sách 28 đặc trưng.

feature_engineering.py
    Tạo 9 đặc trưng từ 19 đặc trưng tài chính gốc.

predict.py
    Hàm dự đoán cho website.

sample_customers.csv
    Ba hồ sơ khách hàng mẫu.

sample_expected_results.csv
    Kết quả dự đoán tham chiếu cho hồ sơ mẫu.

requirements.txt
    Danh sách thư viện Python cần cài đặt.

---

## 3. Installation

Khuyến nghị sử dụng Python 3.14.6.

Mở Terminal tại thư mục WEB_HANDOVER.

Cài đặt thư viện:

    python -m pip install -r requirements.txt

Đặt các file Python, mô hình và cấu hình
trong cùng thư mục.

---

## 4. Required Input Features

Website cần cung cấp 19 trường dữ liệu:

LIMIT_BAL

PAY_0
PAY_2
PAY_3
PAY_4
PAY_5
PAY_6

BILL_AMT1
BILL_AMT2
BILL_AMT3
BILL_AMT4
BILL_AMT5
BILL_AMT6

PAY_AMT1
PAY_AMT2
PAY_AMT3
PAY_AMT4
PAY_AMT5
PAY_AMT6

Ý nghĩa:

LIMIT_BAL:
    Hạn mức tín dụng.

PAY_0 đến PAY_6:
    Trạng thái trả nợ trong 6 kỳ lịch sử.
    PAY_0 là kỳ gần nhất; không có PAY_1.
    Sử dụng đúng mã trạng thái của bộ dữ liệu UCI.
    Không tự chuyển mã sang số ngày chậm trả.

BILL_AMT1 đến BILL_AMT6:
    Số tiền sao kê của 6 kỳ.

PAY_AMT1 đến PAY_AMT6:
    Số tiền thanh toán của 6 kỳ.

Các giá trị tiền tệ trong bộ dữ liệu gốc
được ghi nhận bằng đô la Đài Loan (NT$).
Không tự thay đổi đơn vị khi dự đoán.

Không yêu cầu người dùng nhập 9 đặc trưng
Feature Engineering.

---

## 5. Predict One Customer

Example:

    import pandas as pd
    from predict import predict_customer

    sample = pd.read_csv("sample_customers.csv")

    customer = sample.iloc[0].drop(
        labels=["Customer_ID"]
    ).to_dict()

    result = predict_customer(customer)

    print(result)

Output structure:

    {
        "default_probability": 0.0913,
        "credit_score": 800,
        "risk_group": "Green",
        "predicted_default": 0
    }

Đây là ví dụ định dạng. Xác suất thực tế
được trả về với độ chính xác đầy đủ.

---

## 6. Predict Multiple Customers

Example:

    import pandas as pd
    from predict import predict_batch

    data = pd.read_csv("sample_customers.csv")

    customer_ids = data["Customer_ID"].copy()

    X = data.drop(columns=["Customer_ID"])

    results = predict_batch(X)

    results.insert(
        0, "Customer_ID", customer_ids.to_numpy()
    )

    results.to_csv(
        "prediction_results.csv",
        index=False
    )

Website có thể sử dụng hàm này để dự đoán
nhiều khách hàng từ file CSV hoặc Excel.

---

## 7. Prediction Output

default_probability:
    Xác suất default do mô hình dự đoán.

credit_score:
    Điểm tín dụng nội bộ từ 300 đến 850.

risk_group:
    Green: Probability < 0.15

    Yellow: 0.15 <= Probability < 0.25

    Red: Probability >= 0.25

predicted_default:
    0: Probability < 0.25

    1: Probability >= 0.25

Credit Score được tính bằng công thức:

    round(300 + 550 * (1 - probability))

Trong code sử dụng numpy.rint để làm tròn.

Ngưỡng Green/Yellow/Red là quy tắc minh họa
của dự án, chưa phải chính sách ngân hàng.

---

## 8. Website Integration

Đề xuất giao diện gồm:

1. Chọn khách hàng mẫu.
2. Nhập hoặc chỉnh sửa thông tin tài chính.
3. Tải CSV/Excel để đánh giá hàng loạt.
4. Hiển thị kết quả dự đoán.
5. Xuất kết quả ra CSV.

Luồng xử lý:

19 Original Features
    |
    v
feature_engineering.py
    |
    v
28 Model Features
    |
    v
credit_default_hgb.joblib
    |
    v
Default Probability
    |
    v
Credit Score + Risk Group + Predicted Default

Không cần huấn luyện lại mô hình.

---

## 9. Important Notes

- Không tự thay đổi công thức Feature Engineering.
- Không thay đổi thứ tự 28 đặc trưng.
- Không tự điền 0 cho dữ liệu bị thiếu.
- LIMIT_BAL phải lớn hơn 0.
- Không dùng Actual_Default làm đầu vào.
- Không sử dụng Customer_ID làm đặc trưng.
- Không thay đổi đơn vị tiền tệ.
- Không tự thay đổi ngưỡng phân loại.
- Cần kiểm tra mã trạng thái trả nợ hợp lệ.
- Chỉ tải mô hình joblib từ nguồn tin cậy.
- Không tải dữ liệu khách hàng thật lên dịch vụ công khai.

Đây là mô hình học thuật dựa trên dữ liệu
lịch sử UCI Default of Credit Card Clients.

Không sử dụng kết quả để tự động phê duyệt
hoặc từ chối tín dụng trong thực tế.

---

## 10. Validation

Feature Engineering:
    126000 / 126000 values matched.

Model Probability:
    Maximum difference = 0.0.

Batch Prediction:
    4500 test customers processed.

Sample Customers:
    3 demo customers validated.

Các kết quả trên xác nhận tính nhất quán
trên dữ liệu kiểm thử, không bảo đảm hiệu quả
trên dữ liệu vận hành mới.
