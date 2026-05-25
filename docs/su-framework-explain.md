# SU Framework — Giải thích Pipeline chi tiết & Phân tích Performance

> **Paper gốc**: *"Predicting running time of aerodynamic jobs in HPC system by combining supervised and unsupervised learning method"*  
> **Tác giả**: Hao Wang, Yi-Qin Dai, Jie Yu, Yong Dong (2021)  
> **DOI**: https://doi.org/10.1186/s42774-021-00077-8

---

## 1. Tổng quan Pipeline

SU Framework kết hợp **Supervised Learning** (Học có giám sát) và **Unsupervised Learning** (Học không giám sát) để dự đoán thời gian chạy (runtime) của các job trên hệ thống HPC. Pipeline gồm 3 giai đoạn chính:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SU FRAMEWORK                             │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  PHASE 1     │    │  PHASE 2     │    │  PHASE 3         │  │
│  │  Clustering  │───>│  KNN Search  │───>│  Local SVR/XGB   │  │
│  │  (k-means++) │    │  (k nearest) │    │  Prediction      │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│   Unsupervised         Unsupervised         Supervised          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết từng Phase

### Phase 1: Unsupervised Clustering (Phân cụm không giám sát)

**Mục tiêu**: Nhóm các job tương tự nhau trong lịch sử lại thành các cluster (cụm).

**Các bước thực hiện**:

1. **Tạo chuỗi User_Job**: Kết hợp `user_id` và `executable_id` (tên job) thành một chuỗi duy nhất cho mỗi job.
   ```
   User_Job = str(user_id) + "_" + str(executable_id)
   Ví dụ: "42_1087" nghĩa là user 42 chạy job có executable_id = 1087
   ```

2. **Tính ma trận Levenshtein Distance (LD)**: Cho mỗi user, tính khoảng cách chỉnh sửa (edit distance) giữa tất cả các cặp chuỗi `User_Job` của user đó. Đây là số bước tối thiểu (thêm, xóa, thay ký tự) để biến chuỗi A thành chuỗi B.
   ```
   Ví dụ: LD("42_1087", "42_1088") = 1  (chỉ cần thay '7' → '8')
          LD("42_1087", "42_9999") = 4  (khác nhiều ký tự)
   ```

3. **Clustering bằng k-means++**: Sử dụng ma trận LD làm input, phân cụm các job của mỗi user riêng biệt.
   - Số cluster `k` được **tự động tối ưu** bằng chỉ số **Calinski-Harabasz Score** (đo mức độ phân tách giữa các cụm).
   - Kết quả: Mỗi job trong lịch sử được gán một `cluster_id`.

**Output Phase 1**: Các tập job tương tự S'₁, S'₂, ... cho mỗi user.

---

### Phase 2: K-Nearest Neighbors (Tìm k láng giềng gần nhất)

**Mục tiêu**: Với mỗi job cần dự đoán, tìm k job tương tự nhất trong cluster tương ứng.

**Các bước thực hiện**:

1. **Xác định cluster**: Tìm xem job cần predict thuộc cluster nào (dựa trên `User_Job` string và bảng mapping từ Phase 1).

2. **Mã hóa đặc trưng**:
   - `CPU_req` (số CPU yêu cầu) → chuẩn hóa StandardScaler
   - `requested_time` (thời gian ước tính bởi user) → chuẩn hóa StandardScaler
   - `Submit_time` → mã hóa vòng tròn (cyclic encoding):
     ```
     hours = (submit_time % 86400) / 3600  # Chuyển sang giờ trong ngày
     Submit_sin = sin(2π × hours / 24)
     Submit_cos = cos(2π × hours / 24)
     ```
     Lý do: Thời gian submit có tính chu kỳ (23:59 gần với 00:01), nên dùng sin/cos để mã hóa đúng.

3. **Tính khoảng cách Euclidean** (Công thức 4 trong paper):
   ```
   S_score = √(|cᵢ - cⱼ|² + |rtᵢ - rtⱼ|² + |ssᵢ - ssⱼ|² + |scᵢ - scⱼ|²)
   ```
   Trong đó: c = CPU_req, rt = requested_time, ss = Submit_sin, sc = Submit_cos (tất cả đã chuẩn hóa).

4. **Chọn k job gần nhất**: Sắp xếp theo `S_score` tăng dần, lấy k job đầu tiên.

**Xử lý trường hợp đặc biệt (Fallback)**:
- Nếu cluster quá nhỏ (< k jobs) → mở rộng sang tất cả job của user đó.
- Nếu user chưa từng xuất hiện (new user) → tìm k neighbors trong toàn bộ lịch sử.

**Output Phase 2**: Tập k job tương tự S'' dùng làm dữ liệu huấn luyện cho Phase 3.

---

### Phase 3: Supervised Prediction (Dự đoán có giám sát)

**Mục tiêu**: Huấn luyện model local trên tập S'' để dự đoán runtime.

**Các bước thực hiện**:

1. **Huấn luyện model local**: Sử dụng k job trong S'' làm training data.
   - **Input features**: `(CPU_req, requested_time, Submit_sin, Submit_cos)` (đã chuẩn hóa)
   - **Target**: `run_time` (thời gian chạy thực tế)
   - **Model**: XGBoost GPU (cải tiến so với SVR trong paper gốc)

2. **Dự đoán**: Đưa features của job cần predict vào model local → nhận được `Predicted_time`.

3. **Điều chỉnh bằng hệ số α** (Công thức 5 trong paper):
   ```
   Final = α × Predict(job_new)
   ```
   - `α > 1`: Tăng giá trị dự đoán lên → giảm tỉ lệ underestimation (đánh giá thấp).
   - `α` được tự động tối ưu trên tập validation để đạt APA cao nhất.

---

## 3. Các chỉ số đánh giá

| Chỉ số | Công thức | Ý nghĩa |
|--------|-----------|----------|
| **APA** (Average Predictive Accuracy) | `APA_i = min(Pred_i, Actual_i) / max(Pred_i, Actual_i)` | Trung bình độ chính xác dự đoán. Giá trị [0, 1], càng gần 1 càng tốt. |
| **MAE** (Mean Absolute Error) | `MAE = mean(\|Pred - Actual\|)` | Sai số tuyệt đối trung bình (đơn vị: giây). Càng thấp càng tốt. |
| **UR** (Underestimation Rate) | `UR = count(Pred < Actual) / n` | Tỉ lệ đánh giá thấp. Quan trọng vì underestimation khiến job bị kill. |
| **R²** (R-squared) | Hệ số xác định | Tỉ lệ phương sai được giải thích. Giá trị [-∞, 1], > 0 nghĩa là model tốt hơn đoán trung bình. |

---

## 4. Kết quả Performance

### 4.1 Trên dataset ANL (ANL-Intrepid-2009)

```
==================================================
SU Framework Test Results — ANL Dataset
==================================================
MAE:                  2336.4176 s (~39 phút)
RMSE:                 6297.8940 s
R² Score:             0.5419
Underestimation Rate: 16.21%
Average Pred Accuracy:0.7633
Inference Time:       0.043340 s/sample
==================================================
```

**Nhận xét**: Performance khá tốt. APA đạt 76.33% (gần với 80.46% của paper gốc trên dataset CARDC). R² = 0.54 cho thấy model giải thích được hơn 54% phương sai của runtime.

### 4.2 Trên dataset HCMUT (HCMUT-SuperNodeXP-2017)

```
==================================================
SU Framework Test Results — HCMUT Dataset
==================================================
MAE:                  66264.8330 s (~18.4 giờ)
RMSE:                 136306.3255 s
R² Score:             -0.3489
Underestimation Rate: 27.33%
Average Pred Accuracy:0.5785
Inference Time:       0.046349 s/sample
==================================================
```

**Nhận xét**: Performance rất tệ. R² âm (-0.35) nghĩa là model **tệ hơn cả việc đoán bằng giá trị trung bình**. MAE = 18.4 giờ là sai số không chấp nhận được.

---

## 5. Phân tích nguyên nhân — Tại sao SU Framework không phù hợp?

### 5.1 So sánh đặc tính 2 dataset

| Đặc tính | ANL | HCMUT | Tác động |
|----------|-----|-------|----------|
| **Số jobs** | 50,498 | 10,131 | HCMUT có ít data hơn 5 lần → cluster nhỏ hơn, KNN kém tin cậy hơn |
| **Số unique users** | 208 | **10** | HCMUT chỉ có 10 users → clustering per-user có rất ít sự đa dạng |
| **Số unique executables** | **1** | **1,440** | ANL chỉ có 1 loại job (đồng nhất). HCMUT có 1,440 loại job khác nhau (cực kỳ đa dạng) |
| **Runtime trung bình** | 1.9 giờ | **20.5 giờ** | HCMUT có jobs lớn hơn 10 lần → biên độ sai lệch lớn hơn |
| **Runtime max** | 113 giờ | **712 giờ** | HCMUT có outliers cực lớn |
| **Hệ số biến thiên (CV)** | 1.43 | **2.20** | HCMUT phân tán hơn nhiều → khó dự đoán hơn |
| **Corr(requested_time, run_time)** | **0.7191** | **0.0150** | 🔴 **Đây là nguyên nhân chính** |
| **Jobs/user trung bình** | 242.8 | 1,013.1 | HCMUT có nhiều jobs/user nhưng loại job quá đa dạng |

### 5.2 Phân tích chi tiết nguyên nhân

#### 🔴 Nguyên nhân 1: `requested_time` vô giá trị trên HCMUT

Đây là nguyên nhân **quan trọng nhất**. 

- **ANL**: Tương quan giữa `requested_time` và `run_time` là **0.72** (mạnh). Nghĩa là khi user ước tính job chạy lâu thì thực tế nó cũng chạy lâu → feature này cực kỳ hữu ích cho prediction.
- **HCMUT**: Tương quan chỉ là **0.015** (gần bằng 0). Nghĩa là user ở HCMUT đưa ra ước tính thời gian **hoàn toàn ngẫu nhiên**, không liên quan gì đến thời gian chạy thực → feature này trở thành noise (nhiễu).

Paper gốc lưu ý rằng "the estimated time given by users of more than half of the jobs is five times or more than the actual running time" — điều này đúng với HCMUT nhưng ở mức nghiêm trọng hơn: user không chỉ overestimate mà còn estimate **không có pattern** nào cả.

#### 🟠 Nguyên nhân 2: Quá ít users + quá nhiều loại job

SU Framework dựa trên giả định **"same user tends to submit similar jobs"** (cùng user thường submit job tương tự). Giả định này:
- **Đúng với ANL**: 208 users, chỉ 1 loại executable → mỗi user chạy đi chạy lại cùng 1 loại job → clustering hiệu quả.
- **Sai với HCMUT**: 10 users nhưng 1,440 loại executable → mỗi user chạy hàng trăm loại job khác nhau → clustering không thể nhóm job tương tự lại được.

#### 🟡 Nguyên nhân 3: Biến động runtime quá lớn

- **ANL**: CV (Coefficient of Variation) = 1.43, runtime trung bình 1.9 giờ → biên độ dao động vừa phải.
- **HCMUT**: CV = 2.20, runtime trung bình 20.5 giờ, max 712 giờ → biên độ dao động cực lớn.

Với mỗi job trên HCMUT, runtime có thể dao động từ 10 phút đến 30 ngày. Khi local SVR/XGBoost chỉ huấn luyện trên k=50 neighbors, mẫu quá ít để nắm bắt sự biến động khổng lồ này.

#### 🟡 Nguyên nhân 4: Feature engineering quá đơn giản

SU Framework chỉ sử dụng 4 features:
- `requested_processors` (số CPU yêu cầu)
- `requested_time` (thời gian ước tính — vô dụng trên HCMUT)
- `submit_sin`, `submit_cos` (thời điểm submit trong ngày)

Các feature quan trọng khác như `used_memory`, `wait_time`, `group_id`, `queue_id` không được sử dụng → model thiếu thông tin để phân biệt các job.

---

## 6. Kết luận

### SU Framework phù hợp khi:
- ✅ Workload **đồng nhất** (ít loại job, tập trung vào 1 lĩnh vực như aerodynamics)
- ✅ Users đưa ra **ước tính thời gian hợp lý** (requested_time có tương quan cao với run_time)
- ✅ Số lượng users **đủ lớn** để clustering có ý nghĩa
- ✅ Runtime có **biên độ dao động vừa phải**

### SU Framework KHÔNG phù hợp khi:
- ❌ Workload **đa dạng** (nhiều loại job khác nhau — HCMUT có 1,440 loại)
- ❌ Users **ước tính thời gian bừa bãi** (corr ≈ 0 — như trên HCMUT)
- ❌ Runtime có **biến động cực lớn** (CV > 2, max/min ratio > 1000x)
- ❌ Số users quá ít để clustering per-user có ý nghĩa thống kê

### So sánh với paper gốc (CARDC dataset):

| Chỉ số | Paper (CARDC) | ANL | HCMUT |
|--------|--------------|-----|-------|
| APA | **0.8046** | 0.7633 | 0.5785 |
| MAE | 2762.7 s | 2336.4 s | 66264.8 s |
| UR | 24.85% | 16.21% | 27.33% |

- **ANL** đạt performance gần bằng paper gốc vì có đặc tính tương tự CARDC (workload đồng nhất, requested_time có ý nghĩa).
- **HCMUT** performance rất thấp vì vi phạm các giả định cốt lõi của SU Framework.

> **Phát hiện quan trọng**: Hiệu quả của SU Framework phụ thuộc rất lớn vào **chất lượng thông tin ước tính thời gian từ phía user** (`requested_time`). Khi thông tin này thiếu chính xác, toàn bộ pipeline trở nên vô nghĩa vì feature quan trọng nhất bị biến thành noise.
