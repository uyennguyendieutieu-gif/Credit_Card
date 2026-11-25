# Kịch bản video giải thích code — CreditAI

Tổng thời lượng gợi ý: **~12–14 phút** (9 cảnh). Có thể cắt bớt phần nào nếu giới hạn thời gian của đề.

## Chuẩn bị trước khi quay (làm 1 lần)

Mở sẵn các tab này theo đúng thứ tự để không phải tìm giữa chừng:

1. VS Code (hoặc Jupyter) mở thư mục `WEB_HANDOVER`, mở sẵn 5 file trong `notebooks/` theo thứ tự 01→05
2. Trình duyệt tab 1: `README.md` (xem dạng preview)
3. Trình duyệt tab 2: `reports/preprocessing_report.html` (mở trực tiếp từ máy)
4. Trình duyệt tab 3: `reports/test_evaluation_report.html` (mở trực tiếp từ máy)
5. Trình duyệt tab 4: `https://creditcard-demo.streamlit.app/` (web demo đã deploy)
6. Terminal mở sẵn tại thư mục dự án (phòng khi cần chạy lệnh)

Chạy thử notebook trước 1 lần để các ô code đã có output sẵn (tránh phải chờ chạy khi quay).

---

## Cảnh 1 — Mở đầu & bài toán kinh doanh (~1 phút)

**Mở file:** `README.md` (phần đầu, mục "Project Overview")

**Mục đích:** Cho người xem hiểu *vì sao* làm dự án này trước khi xem *làm như thế nào*.

**Nói gì:**
> "Nhóm mình làm CreditAI — hệ thống hỗ trợ nhân viên ngân hàng đánh giá rủi ro hồ sơ vay. Bài toán kinh doanh là: ngân hàng nhận hàng nghìn hồ sơ vay mỗi ngày, xét duyệt thủ công vừa chậm vừa dễ sai lệch giữa các nhân viên. Mình dùng machine learning để dự đoán xác suất một khách hàng sẽ vỡ nợ ở kỳ thanh toán tiếp theo, từ đó tự động phân khách hàng thành 3 nhóm màu: Green — duyệt ngay, Yellow — cần thẩm định thêm, Red — từ chối."

**Hành động:** Lướt qua phần "Output" trong README để chỉ 4 thứ output: Default Probability, Credit Score, Risk Group, Predicted Default.

---

## Cảnh 2 — Dữ liệu & khám phá dữ liệu (EDA) (~1.5 phút)

**Mở file:** `notebooks/01_EDA.ipynb`

**Mục đích:** Chứng minh nhóm hiểu dữ liệu trước khi xử lý, không "nhắm mắt" đưa vào model.

**Nói gì:**
> "Dữ liệu dùng là bộ UCI Default of Credit Card Clients — 30.000 khách hàng thẻ tín dụng ở Đài Loan, có lịch sử trả nợ 6 tháng. Tỷ lệ vỡ nợ khoảng 22%, nghĩa là dữ liệu mất cân bằng lớp khá rõ."

**Hành động trên màn hình:**
- Cuộn tới Cell 2–3: chỉ bảng kiểm tra chất lượng dữ liệu — nói "không có giá trị thiếu, không có dòng trùng thật sự vì mình kiểm tra cả cột ID"
- Cuộn tới Cell 6: biểu đồ "Default Rate by PAY_0" — nói: "Trễ hạn càng nhiều tháng thì tỷ lệ vỡ nợ càng tăng mạnh, đây là tín hiệu rõ nhất trong toàn bộ dữ liệu"
- Cuộn tới Cell 13: bảng tổng kết "Danh sách vấn đề tiền xử lý" — nói: "Đây là checklist nhóm mình lập ra từ EDA để quyết định bước tiền xử lý tiếp theo"

---

## Cảnh 3 — Tiền xử lý & Feature Engineering (~2 phút)

**Mở file:** `notebooks/02_Preprocessing.ipynb`, sau đó chuyển sang `feature_engineering.py`

**Mục đích:** Đây là phần kỹ thuật quan trọng nhất để giám khảo thấy nhóm hiểu sâu, không chỉ gọi hàm có sẵn.

**Nói gì (mở đầu):**
> "Đầu tiên mình chia dữ liệu theo tỷ lệ 70% Train, 15% Validation, 15% Test, dùng stratify để giữ nguyên tỷ lệ vỡ nợ ở cả 3 tập. Tập Test được giữ riêng hoàn toàn, chỉ dùng đúng 1 lần ở cuối để tránh rò rỉ thông tin."

**Hành động:** Chỉ Cell 1 — đoạn `train_test_split(..., stratify=y, random_state=42)`.

**Nói gì (feature engineering):**
> "Từ 19 đặc trưng tài chính gốc — gồm hạn mức, 6 kỳ trạng thái trả nợ, 6 kỳ dư nợ sao kê, 6 kỳ số tiền đã trả — mình tạo thêm 9 đặc trưng mới để mô hình nhìn ra rủi ro rõ hơn. Ví dụ UTILIZATION_RATIO là tỷ lệ dùng hạn mức, lấy dư nợ sao kê chia cho hạn mức tín dụng."

**Hành động:** Mở `feature_engineering.py`, chỉ hàm `create_features()` — đọc lướt qua 4 nhóm đặc trưng: lịch sử thanh toán, số tiền thanh toán, số tiền sao kê, tỷ lệ sử dụng.

**Nói gì (kết):**
> "Sau khi kiểm tra, nhóm phát hiện 1 đặc trưng dư thừa — AVG_PAYMENT_6M luôn bằng TOTAL_PAYMENT_6M chia 6 — nên loại bỏ. Kết quả cuối: 19 cộng 9 trừ 1 bằng 28 đặc trưng đưa vào mô hình."

*(Tuỳ chọn: mở thêm `reports/preprocessing_report.html` trên trình duyệt 10–15 giây để chỉ bảng tổng hợp 28 đặc trưng cho trực quan hơn.)*

---

## Cảnh 4 — Huấn luyện & tinh chỉnh mô hình (~2.5 phút)

**Mở file:** `notebooks/03_ModelTraining.ipynb`

**Mục đích:** Đây là phần "nặng" nhất — cần cho thấy nhóm không chỉ chọn 1 mô hình ngẫu nhiên mà có quy trình so sánh bài bản.

**Nói gì:**
> "Nhóm huấn luyện 3 mô hình cơ sở: Logistic Regression — đơn giản, dễ giải thích, cần chuẩn hóa dữ liệu; Random Forest — nhiều cây quyết định bỏ phiếu; và Hist Gradient Boosting — các cây nối tiếp, cây sau sửa lỗi cây trước."

**Hành động:** Lướt nhanh Cell 2, 4, 6 — chỉ đoạn cấu hình từng mô hình.

**Nói gì (ngưỡng quyết định):**
> "Vì dữ liệu mất cân bằng, dùng ngưỡng 0,5 mặc định sẽ bỏ sót nhiều khách xấu. Nhóm quét một lưới ngưỡng từ 0,1 đến 0,8, chọn ngưỡng cho F1 cao nhất."

**Hành động:** Chỉ Cell 3 — biểu đồ Precision/Recall/F1 theo ngưỡng.

**Nói gì (ablation + tuning):**
> "Để chứng minh 9 đặc trưng mới thật sự có ích, nhóm huấn luyện lại cả 3 mô hình chỉ với 19 đặc trưng gốc rồi so với bộ 28 đặc trưng — Logistic Regression cải thiện rõ nhất, AUC tăng từ 0,711 lên 0,757. Sau đó nhóm dùng RandomizedSearchCV với 3-fold Stratified Cross Validation để tìm siêu tham số tốt nhất, chấm điểm bằng PR-AUC vì dữ liệu mất cân bằng."

**Hành động:** Chỉ Cell 8 (ablation) rồi Cell 9/11 (tuning) — chỉ đoạn `param_distributions` và `RandomizedSearchCV(...)`.

**Nói gì (chọn mô hình cuối):**
> "Mô hình thắng là Hist Gradient Boosting sau khi tinh chỉnh, với ROC-AUC 0,7812 trên Validation. Mô hình được lưu thành file `credit_default_hgb.joblib`."

**Hành động:** Chỉ Cell 13 — đoạn `joblib.dump(...)`.

---

## Cảnh 5 — Đánh giá độc lập trên tập Test (~2 phút)

**Mở file:** `notebooks/04_ModelEvaluation.ipynb`, sau đó `reports/test_evaluation_report.html`

**Mục đích:** Chứng minh kết quả không phải "ăn may" trên dữ liệu đã thấy — đây là tập Test mô hình chưa từng học qua.

**Nói gì:**
> "Đây là bước quan trọng nhất để đảm bảo kết quả đáng tin: mình chấm điểm mô hình trên 4.500 khách hàng ở tập Test mà nó chưa từng thấy trong lúc huấn luyện hay tinh chỉnh."

**Hành động:** Chỉ Cell 2 — bảng metrics và confusion matrix.

**Nói gì:**
> "Kết quả: AUC 0,78, KS 0,43 — hai chỉ số này được ngành tín dụng dùng phổ biến để đo khả năng phân biệt khách tốt và khách xấu. Trong 996 khách thực sự vỡ nợ, mô hình bắt đúng 607, tức Recall khoảng 61%."

**Hành động:** Chuyển sang tab `reports/test_evaluation_report.html` trên trình duyệt — chỉ lần lượt: bảng so sánh 3 mô hình, biểu đồ ROC, biểu đồ feature importance (nói: "PAY_0 — trạng thái trả nợ kỳ gần nhất — là yếu tố ảnh hưởng mạnh nhất, đúng với trực giác nghiệp vụ").

---

## Cảnh 6 — Chấm điểm tín dụng & phân nhóm rủi ro (~1.5 phút)

**Mở file:** `notebooks/05_CreditScoring.ipynb`, sau đó `predict.py`

**Mục đích:** Đây là phần biến xác suất kỹ thuật thành thứ nhân viên ngân hàng đọc hiểu được.

**Nói gì:**
> "Xác suất vỡ nợ là một con số từ 0 đến 1, khó hình dung cho người dùng cuối, nên nhóm quy đổi thành điểm tín dụng nội bộ từ 300 đến 850 bằng công thức tuyến tính: 300 cộng 550 nhân với 1 trừ xác suất. Sau đó phân khách hàng thành 3 nhóm: Green nếu xác suất dưới 15%, Yellow từ 15% đến dưới 25%, Red từ 25% trở lên."

**Hành động:** Mở `predict.py`, chỉ hàm `probability_to_score()` và `assign_risk_group()`.

**Nói gì (kiểm chứng):**
> "Để chứng minh cách chia màu này có ý nghĩa thật chứ không phải chia tùy tiện, nhóm đối chiếu với tỷ lệ vỡ nợ thực tế trên tập Test: nhóm Green chỉ 9,2% thực sự vỡ nợ, Yellow 18,6%, còn Red lên tới 48,6% — cao gấp 5 lần Green."

*(Hành động: có thể chỉ bảng risk-group trong `reports/test_evaluation_report.html` nếu chưa chỉ ở Cảnh 5.)*

---

## Cảnh 7 — Đóng gói mô hình cho production (~1 phút)

**Mở file:** `credit_default_config.json`, lướt qua `feature_engineering.py` và `predict.py`

**Mục đích:** Cho thấy mô hình không chỉ nằm trong notebook mà đã đóng gói thành sản phẩm dùng lại được.

**Nói gì:**
> "Nhóm đóng gói mô hình thành 4 phần: file mô hình `credit_default_hgb.joblib`, file cấu hình `credit_default_config.json` lưu danh sách 28 đặc trưng và ngưỡng quyết định, module `feature_engineering.py` để tái tạo đúng 28 đặc trưng từ 19 đặc trưng gốc, và module `predict.py` chứa 2 hàm `predict_customer` cho một khách và `predict_batch` cho nhiều khách."

**Hành động:** Mở `credit_default_config.json`, chỉ 3 trường: `feature_columns`, `decision_threshold`, `hyperparameters`.

---

## Cảnh 8 — Demo ứng dụng web (~2 phút)

**Mở:** trình duyệt tab `https://creditcard-demo.streamlit.app/` (hoặc `streamlit_app.py` nếu muốn chỉ code trước)

**Mục đích:** Phần "wow" cuối — biến mô hình thành sản phẩm ai cũng dùng được, không cần biết code.

**Nói gì (giới thiệu):**
> "Để mô hình thực sự dùng được, nhóm xây dựng web demo bằng Streamlit và deploy công khai, không cần cài đặt gì."

**Hành động — demo trực tiếp (quan trọng nhất của cảnh này):**
1. Ở sidebar, chỉ phần chú giải màu Green/Yellow/Red
2. Chọn "DEMO-003 (rủi ro cao)" ở ô chọn mẫu → bấm **Dự đoán** → chỉ kết quả: điểm 619, vòng tròn gauge màu đỏ, nhãn "Red"
3. Nói: "Đây là một hồ sơ rủi ro cao — trễ hạn 2 tháng liên tiếp, hạn mức thấp — model cho xác suất vỡ nợ gần 42%, rơi vào nhóm Red"
4. Chuyển qua tab "Dự đoán hàng loạt" → upload thử `samples/sample_customers.csv` → bấm Dự đoán hàng loạt → chỉ bảng kết quả có tô màu theo nhóm rủi ro
5. Bấm nút Tải CSV kết quả

**Hành động (tuỳ chọn, nếu còn thời gian):** Mở nhanh `streamlit_app.py`, chỉ dòng `from predict import predict_customer, predict_batch` — nói: "Web chỉ gọi lại đúng hàm đã kiểm chứng ở bước đóng gói, không viết lại logic dự đoán lần nữa, nên kết quả luôn khớp 100% với mô hình gốc."

---

## Cảnh 9 — Hạn chế & kết luận (~1 phút)

**Không cần mở file** — nói trực tiếp vào camera hoặc slide tổng kết.

**Nói gì:**
> "Một vài hạn chế nhóm mình nhận thấy: dữ liệu là khách hàng thẻ tín dụng Đài Loan năm 2005, không phải khoản vay tiêu dùng tại Việt Nam hiện tại, nên cần huấn luyện lại với dữ liệu thật nếu áp dụng thực tế. Mô hình còn bỏ sót khoảng 39% khách vỡ nợ, và ngưỡng phân nhóm màu hiện tại là quy tắc minh họa, chưa dựa trên chi phí thật của doanh nghiệp.
>
> Tổng kết lại, nhóm đã xây dựng quy trình hoàn chỉnh từ dữ liệu thô đến ứng dụng web: thu thập, làm sạch, tạo đặc trưng, huấn luyện và so sánh 3 mô hình, tinh chỉnh, đánh giá độc lập trên tập Test, và đóng gói thành sản phẩm demo thực tế. Cảm ơn thầy/cô đã theo dõi."

---

## Mẹo khi quay

- Quay từng cảnh riêng rồi ghép (dễ quay lại nếu lỡ lời, không phải quay lại từ đầu).
- Phần code: zoom to tối thiểu 150% để chữ rõ khi xem trên điện thoại.
- Nói chậm hơn bình thường khoảng 10–15% — lúc quay sẽ thấy nhanh hơn lúc nghe lại.
- Nếu quay máy tính (screen record), tắt thông báo (Notification) trước khi quay.
