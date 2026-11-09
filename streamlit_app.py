# =========================================================
# CREDITAI - CREDIT SCORING DEMO (STREAMLIT)
# Chay local:    streamlit run streamlit_app.py
# Deploy:        push len GitHub, deploy tren share.streamlit.io
# =========================================================

import pandas as pd
import streamlit as st

from predict import predict_customer, predict_batch

st.set_page_config(page_title="CreditAI Scoring", page_icon="\U0001F4B3", layout="wide")

PAY_FIELDS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_FIELDS = [f"BILL_AMT{i}" for i in range(1, 7)]
AMT_FIELDS = [f"PAY_AMT{i}" for i in range(1, 7)]
ALL_FIELDS = ["LIMIT_BAL"] + PAY_FIELDS + BILL_FIELDS + AMT_FIELDS

PAY_OPTIONS = {
    -2: "-2 · Không phát sinh giao dịch",
    -1: "-1 · Trả đúng hạn",
    0: "0 · Xoay vòng, trả tối thiểu",
    1: "1 · Trễ 1 tháng",
    2: "2 · Trễ 2 tháng",
    3: "3 · Trễ 3 tháng",
    4: "4 · Trễ 4 tháng",
    5: "5 · Trễ 5 tháng",
    6: "6 · Trễ 6 tháng",
    7: "7 · Trễ 7 tháng",
    8: "8 · Trễ 8 tháng",
}

SAMPLES = {
    "— Tự nhập —": None,
    "DEMO-001 (an toàn)": {
        "LIMIT_BAL": 110000, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 79646, "BILL_AMT2": 110818, "BILL_AMT3": 107558, "BILL_AMT4": 80437, "BILL_AMT5": 79727, "BILL_AMT6": 81001,
        "PAY_AMT1": 50000, "PAY_AMT2": 3500, "PAY_AMT3": 10000, "PAY_AMT4": 3000, "PAY_AMT5": 3000, "PAY_AMT6": 3000,
    },
    "DEMO-002 (trung bình)": {
        "LIMIT_BAL": 360000, "PAY_0": 1, "PAY_2": -2, "PAY_3": -2, "PAY_4": -1, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 0, "BILL_AMT2": 1680, "BILL_AMT3": 0, "BILL_AMT4": 5950, "BILL_AMT5": 6352, "BILL_AMT6": 6563,
        "PAY_AMT1": 1680, "PAY_AMT2": 0, "PAY_AMT3": 5950, "PAY_AMT4": 500, "PAY_AMT5": 317, "PAY_AMT6": 1150,
    },
    "DEMO-003 (rủi ro cao)": {
        "LIMIT_BAL": 50000, "PAY_0": 1, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 17479, "BILL_AMT2": 18195, "BILL_AMT3": 17614, "BILL_AMT4": 18630, "BILL_AMT5": 19000, "BILL_AMT6": 19421,
        "PAY_AMT1": 1300, "PAY_AMT2": 0, "PAY_AMT3": 1310, "PAY_AMT4": 680, "PAY_AMT5": 727, "PAY_AMT6": 6000,
    },
}

RISK_COLOR = {"Green": "#1f7a3d", "Yellow": "#9a7d12", "Red": "#a3342a"}
RISK_BG = {"Green": "#e1f0e3", "Yellow": "#f6edc9", "Red": "#f6dbd7"}


def render_result(result: dict):
    prob = result["default_probability"]
    score = result["credit_score"]
    group = result["risk_group"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Credit Score (300–850)", score)
    c2.metric("Xác suất vỡ nợ", f"{prob * 100:.2f}%")
    c3.metric("Dự đoán nhãn", "Có nguy cơ" if result["predicted_default"] == 1 else "An toàn")

    st.markdown(
        f"""<div style="display:inline-block;padding:6px 16px;border-radius:999px;
        background:{RISK_BG[group]};color:{RISK_COLOR[group]};font-weight:600;">
        Nhóm rủi ro: {group}</div>""",
        unsafe_allow_html=True,
    )
    st.progress(min(max(prob, 0.0), 1.0))


st.title("💳 CreditAI — Dự đoán rủi ro vỡ nợ khách hàng")
st.caption(
    "Nhập 19 thông tin tài chính gốc (đơn vị NT$ — đô la Đài Loan), hệ thống tự tính thêm "
    "đặc trưng và chấm điểm bằng mô hình Hist Gradient Boosting đã huấn luyện."
)

tab_single, tab_batch = st.tabs(["Dự đoán 1 khách hàng", "Dự đoán hàng loạt (CSV)"])

# ---------------------------------------------------------
# TAB 1: DU DOAN 1 KHACH HANG
# ---------------------------------------------------------
with tab_single:
    sample_name = st.selectbox("Chọn khách hàng mẫu", list(SAMPLES.keys()))
    preset = SAMPLES[sample_name] or SAMPLES["DEMO-001 (an toàn)"]

    with st.form("predict_form"):
        st.markdown("**Hạn mức tín dụng**")
        limit_bal = st.number_input("LIMIT_BAL (NT$, phải > 0)", min_value=1, value=int(preset["LIMIT_BAL"]), step=1000,
                                     key=f"single_{sample_name}_LIMIT_BAL")

        st.markdown("**Lịch sử trả nợ 6 kỳ gần nhất (PAY_0 = gần nhất)**")
        pay_cols = st.columns(6)
        pay_values = {}
        for col, field in zip(pay_cols, PAY_FIELDS):
            default_idx = list(PAY_OPTIONS.keys()).index(preset[field])
            pay_values[field] = col.selectbox(field, list(PAY_OPTIONS.keys()), index=default_idx,
                                               format_func=lambda v: PAY_OPTIONS[v], key=f"single_{sample_name}_{field}")

        st.markdown("**Dư nợ sao kê 6 kỳ (BILL_AMT, có thể âm)**")
        bill_cols = st.columns(6)
        bill_values = {}
        for col, field in zip(bill_cols, BILL_FIELDS):
            bill_values[field] = col.number_input(field, value=int(preset[field]), step=100, key=f"single_{sample_name}_{field}")

        st.markdown("**Số tiền đã trả 6 kỳ (PAY_AMT, ≥ 0)**")
        amt_cols = st.columns(6)
        amt_values = {}
        for col, field in zip(amt_cols, AMT_FIELDS):
            amt_values[field] = col.number_input(field, min_value=0, value=int(preset[field]), step=100, key=f"single_{sample_name}_{field}")

        submitted = st.form_submit_button("Dự đoán", type="primary")

    if submitted:
        customer = {"LIMIT_BAL": limit_bal, **pay_values, **bill_values, **amt_values}
        try:
            result = predict_customer(customer)
            render_result(result)
        except Exception as exc:
            st.error(f"Lỗi: {exc}")

# ---------------------------------------------------------
# TAB 2: DU DOAN HANG LOAT
# ---------------------------------------------------------
with tab_batch:
    st.markdown(
        "File CSV cần các cột: `Customer_ID` (tuỳ chọn), `LIMIT_BAL`, `PAY_0`, `PAY_2`–`PAY_6`, "
        "`BILL_AMT1`–`BILL_AMT6`, `PAY_AMT1`–`PAY_AMT6`."
    )
    uploaded = st.file_uploader("Chọn file CSV", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            st.write("Xem trước dữ liệu:")
            st.dataframe(raw_df.head(), use_container_width=True)

            missing = [f for f in ALL_FIELDS if f not in raw_df.columns]
            if missing:
                st.error("File CSV thiếu cột: " + ", ".join(missing))
            elif st.button("Dự đoán hàng loạt", type="primary"):
                ids = raw_df["Customer_ID"] if "Customer_ID" in raw_df.columns else None
                X = raw_df[ALL_FIELDS].copy()

                results_df = predict_batch(X)
                if ids is not None:
                    results_df.insert(0, "Customer_ID", ids.to_numpy())

                st.success(f"Đã dự đoán xong {len(results_df)} khách hàng.")
                st.dataframe(results_df, use_container_width=True)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("Tải CSV kết quả", data=csv_bytes,
                                    file_name="prediction_results.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Lỗi: {exc}")

st.divider()
st.caption(
    "CreditAI — mô hình học thuật dựa trên dữ liệu UCI Default of Credit Card Clients. "
    "Không dùng để tự động phê duyệt tín dụng trong thực tế."
)
