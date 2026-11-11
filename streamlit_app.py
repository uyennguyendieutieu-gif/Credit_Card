# =========================================================
# CREDITAI - CREDIT SCORING DEMO (STREAMLIT)
# Chay local:    streamlit run streamlit_app.py
# Deploy:        push len GitHub, deploy tren share.streamlit.io
# =========================================================

import pandas as pd
import streamlit as st

from predict import predict_customer, predict_batch

st.set_page_config(page_title="CreditAI Scoring", page_icon="\U0001F4B3", layout="wide")

# ---------------------------------------------------------
# GIAO DIEN: font, mau sac, bo vien mac dinh cua Streamlit
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 600 !important; }
code, .stCodeBlock, [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.hero {
    background: linear-gradient(120deg, #0f6e63 0%, #134d45 100%);
    color: #f3f7f5; border-radius: 16px; padding: 28px 32px; margin-bottom: 22px;
}
.hero .eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: .72rem; letter-spacing: .14em;
    text-transform: uppercase; color: #9fe0d4; font-weight: 600; margin-bottom: 8px;
}
.hero h1 { color: #fff !important; font-size: 1.9rem !important; margin: 0 !important; }
.hero p { color: #d7ece7; margin-top: 8px; max-width: 62ch; font-size: .95rem; }

.legend-row { display: flex; gap: 10px; margin-top: 14px; flex-wrap: wrap; }
.legend-pill {
    font-size: .76rem; font-weight: 600; padding: 4px 11px; border-radius: 999px;
    display: inline-flex; align-items: center; gap: 6px;
}

.score-gauge { width: 170px; height: 170px; margin: 6px auto 4px; border-radius: 50%; }
.score-gauge-inner {
    width: 76%; height: 76%; border-radius: 50%; background: #ffffff;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.score-gauge-inner .num { font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.1rem; color: #132725; line-height: 1; }
.score-gauge-inner .den { font-size: .72rem; color: #49605c; margin-top: 2px; }

.risk-pill {
    display: inline-flex; align-items: center; gap: 6px; font-weight: 700; font-size: .92rem;
    padding: 6px 18px; border-radius: 999px; margin: 10px auto 0; width: fit-content;
}

div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
    border-radius: 9px !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

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
RISK_LABEL_VI = {"Green": "Duyệt ngay", "Yellow": "Cần thẩm định thêm", "Red": "Từ chối"}

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 💳 CreditAI")
    st.caption("Hệ thống chấm điểm rủi ro vỡ nợ — mô hình Hist Gradient Boosting.")
    st.divider()
    st.markdown("**Chú giải nhóm rủi ro**")
    for g in ["Green", "Yellow", "Red"]:
        st.markdown(
            f"""<div class="legend-pill" style="background:{RISK_BG[g]};color:{RISK_COLOR[g]};">
            ● {g} — {RISK_LABEL_VI[g]}</div>""",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Mô hình học thuật, dựa trên dữ liệu UCI Default of Credit Card Clients — không dùng để tự động phê duyệt tín dụng trong thực tế.")

# ---------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">CreditAI · Credit Scoring Tool</div>
    <h1>Dự đoán rủi ro vỡ nợ khách hàng</h1>
    <p>Nhập 19 thông tin tài chính gốc (đơn vị NT$ — đô la Đài Loan), hệ thống tự tính thêm
    9 đặc trưng và chấm điểm bằng mô hình Hist Gradient Boosting đã huấn luyện, tinh chỉnh trên
    30.000 khách hàng.</p>
</div>
""", unsafe_allow_html=True)


def render_score_gauge(score: int, group: str):
    pct = max(0, min(100, (score - 300) / (850 - 300) * 100))
    color = RISK_COLOR[group]
    st.markdown(f"""
    <div class="score-gauge" style="background: conic-gradient({color} {pct}%, #e4e9e3 {pct}% 100%);">
        <div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;">
            <div class="score-gauge-inner">
                <div class="num">{score}</div>
                <div class="den">/ 850</div>
            </div>
        </div>
    </div>
    <div style="text-align:center;">
        <span class="risk-pill" style="background:{RISK_BG[group]};color:{color};">● {group} · {RISK_LABEL_VI[group]}</span>
    </div>
    """, unsafe_allow_html=True)


def render_result(result: dict):
    prob = result["default_probability"]
    score = result["credit_score"]
    group = result["risk_group"]

    render_score_gauge(score, group)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Xác suất vỡ nợ", f"{prob * 100:.2f}%")
    c2.metric("Dự đoán nhãn", "Có nguy cơ" if result["predicted_default"] == 1 else "An toàn")
    st.caption(f"Ngưỡng quyết định: 0.25 · Credit Score = round(300 + 550 × (1 − {prob:.4f}))")


tab_single, tab_batch = st.tabs(["🧾 Dự đoán 1 khách hàng", "📂 Dự đoán hàng loạt (CSV)"])

# ---------------------------------------------------------
# TAB 1: DU DOAN 1 KHACH HANG
# ---------------------------------------------------------
with tab_single:
    col_form, col_result = st.columns([1.35, 1], gap="large")

    with col_form:
        with st.container(border=True):
            sample_name = st.selectbox("Chọn khách hàng mẫu", list(SAMPLES.keys()))
            preset = SAMPLES[sample_name] or SAMPLES["DEMO-001 (an toàn)"]

            with st.form("predict_form"):
                st.markdown("**Hạn mức tín dụng**")
                limit_bal = st.number_input("LIMIT_BAL (NT$, phải > 0)", min_value=1, value=int(preset["LIMIT_BAL"]), step=1000,
                                             key=f"single_{sample_name}_LIMIT_BAL")

                st.markdown("**Lịch sử trả nợ 6 kỳ gần nhất (PAY_0 = gần nhất)**")
                pay_cols = st.columns(3)
                pay_values = {}
                for i, field in enumerate(PAY_FIELDS):
                    default_idx = list(PAY_OPTIONS.keys()).index(preset[field])
                    pay_values[field] = pay_cols[i % 3].selectbox(
                        field, list(PAY_OPTIONS.keys()), index=default_idx,
                        format_func=lambda v: PAY_OPTIONS[v], key=f"single_{sample_name}_{field}")

                st.markdown("**Dư nợ sao kê 6 kỳ (BILL_AMT, có thể âm)**")
                bill_cols = st.columns(3)
                bill_values = {}
                for i, field in enumerate(BILL_FIELDS):
                    bill_values[field] = bill_cols[i % 3].number_input(
                        field, value=int(preset[field]), step=100, key=f"single_{sample_name}_{field}")

                st.markdown("**Số tiền đã trả 6 kỳ (PAY_AMT, ≥ 0)**")
                amt_cols = st.columns(3)
                amt_values = {}
                for i, field in enumerate(AMT_FIELDS):
                    amt_values[field] = amt_cols[i % 3].number_input(
                        field, min_value=0, value=int(preset[field]), step=100, key=f"single_{sample_name}_{field}")

                submitted = st.form_submit_button("Dự đoán", type="primary", use_container_width=True)

    with col_result:
        with st.container(border=True):
            st.markdown("**Kết quả dự đoán**")
            if submitted:
                customer = {"LIMIT_BAL": limit_bal, **pay_values, **bill_values, **amt_values}
                try:
                    result = predict_customer(customer)
                    render_result(result)
                except Exception as exc:
                    st.error(f"Lỗi: {exc}")
            else:
                st.caption("Nhập thông tin bên trái và bấm **Dự đoán** để xem kết quả.")

# ---------------------------------------------------------
# TAB 2: DU DOAN HANG LOAT
# ---------------------------------------------------------
with tab_batch:
    with st.container(border=True):
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

                    def _highlight_risk(row):
                        color = RISK_BG.get(row["Risk_Group"], "#ffffff")
                        return [f"background-color: {color}"] * len(row)

                    st.dataframe(results_df.style.apply(_highlight_risk, axis=1), use_container_width=True)

                    csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("⬇️ Tải CSV kết quả", data=csv_bytes,
                                        file_name="prediction_results.csv", mime="text/csv", type="primary")
            except Exception as exc:
                st.error(f"Lỗi: {exc}")

st.divider()
st.caption(
    "CreditAI — mô hình học thuật dựa trên dữ liệu UCI Default of Credit Card Clients. "
    "Không dùng để tự động phê duyệt tín dụng trong thực tế."
)
