


Documents
=
# 💭Mô tả
Folder chứa các thuật toán để biểu diễn văn bản dưới dạng ma trận.
# 📃Cấu trúc folder
```
embedding/
|-- __init__.py
|-- README.md
|-- BoW.py
|-- Tfidf.py
|-- ppmi.py
|-- HellingerPCA.py
|-- word2Vec.py
|-- glove.py
|-- FastText.py
```

## 1️⃣LSA models
Gồm các file: `BoW.py`, `Tfidf.py`, `ppmi.py`
### `BoW.py`
**Lớp `BagOfWords`**: Lớp này xây dựng biểu diễn Bag-of-Words cho các tài liệu văn bản đã làm sạch. Các phương thức chính gồm:
- `__init__`: Khởi tạo với các tham số:
  - `min_word_freq`: tần suất từ tối thiểu  
  - `max_features`: số lượng đặc trưng tối đa  
  - `tokenizer`: bộ tách từ
  
- `fit(documents)`:
  - Tách từ và đếm số lần xuất hiện của mỗi từ trong tập văn bản
  - Lọc các từ theo tần suất tối thiểu và giới hạn số lượng đặc trưng
  - Xây dựng từ điển ánh xạ từ → chỉ số
  
- `transform(documents)`: Chuyển tập tài liệu thành ma trận BoW có kích thước `(n_documents, n_features)`

- `transform_single(document)`: Tạo vector BoW cho một tài liệu đơn lẻ

- `fit_transform(documents)`: Gộp hai bước `fit` và `transform`

- `get_feature_names()`: Trả về danh sách các từ trong từ điển

- `get_vocabulary_size()`: Trả về kích thước của từ điển

- `get_word_frequency(word)`: Trả về số lần xuất hiện của một từ

- `print_vocabulary_info(top_n)`: In thông tin thống kê về từ điển

- `save_vocabulary(filepath)` và `load_vocabulary(filepath)`: Lưu và tải từ điển từ tệp văn bản

**Lớp `SVDModel`**: Lớp thực hiện giảm chiều bằng TruncatedSVD được cài đặt từ đầu bằng phân rã giá trị riêng. Các phương thức bao gồm:
- `__init__`: Khởi tạo với số chiều đầu ra mong muốn (`n_components`)

- `fit(X)`:
  - Tính ma trận hiệp phương sai `X_T X` và phân rã thành các trị riêng
  - Lấy các thành phần chính (`U, S, Vt`) và tỷ lệ phương sai giải thích
  
- `transform(X)`: Chiếu dữ liệu lên không gian giảm chiều

- `fit_transform(X)`: Gộp hai bước `fit` và `transform`

- `inverse_transform(X_transformed)`: Khôi phục dữ liệu từ không gian giảm chiều

**Lớp `BOW_SVD_Embedding`**: Pipeline kết hợp giữa Bag-of-Words và SVD để sinh vector biểu diễn tài liệu. Các phương thức chính gồm:
- `__init__`: Khởi tạo pipeline với tham số tùy chỉnh cho BoW (`bow_args`) và SVD (`dim_reduc_args`)

- `fit(documents)`: Học từ điển và giảm chiều từ tập tài liệu đầu vào

- `transform(documents)`: Chuyển đổi tài liệu sang vector rút gọn đã học

- `fit_transform(documents)`: Gộp hai bước `fit` và `transform`

- `get_feature_names()`: Trả về danh sách từ trong từ điển BoW

- `get_vocabulary_size()`: Trả về kích thước từ điển BoW

---
### `Tfidf.py`
**Lớp `TruncatedSVD`**: Đây là lớp thực hiện xoay trục ma trận sử dụng TruncatedSVD. Các phương thức bao gồm:    
- `fit(X)`: Thực hiện phân rã SVD. Lưu trữ các thành phần chính (`components`), giá trị kỳ dị (`singular values`), và phương sai giải thích (`explained variance`).

- `transform(X)`: Chiếu dữ liệu lên không gian thành phần chính đã học.

- `choose_n_components(threshold)`: Chọn số chiều tối ưu để giữ lại tỷ lệ phương sai mong muốn, mặc định là 95\%.

- `plot_cumulative_variance()`: Vẽ đồ thị biểu diễn tỷ lệ phương sai tích lũy theo số chiều giữ lại.

**Lớp `TEmbedder`**: Lớp này xử lý đầu vào là các tài liệu văn bản và sinh ra biểu diễn vector rút gọn. Các phương thức bao gồm:
- `__init__`: Người dùng có thể tùy chỉnh số chiều đầu ra (`n_components`), số từ tối đa (`max_features`), và lựa chọn chuẩn hóa (`norm`).

- `fit(documents)`: Xây dựng từ điển (dạng term-document}) và tính toán IDF cho từng từ. Sau đó, tính toán TF-IDF và chuẩn hóa, cuối cùng áp dụng `TruncatedSVD` để giảm chiều.

- `transform_doc(documents)`: Chuyển đổi danh sách tài liệu sang vector biểu diễn rút gọn.

- `find_best_n_components(threshold, plot)`: Dựa trên phương sai tích lũy để xác định số chiều tối ưu. Biến `plot` là biến nhị phân, dùng để thực hiện việc vẽ lượng thông tin dựa theo số chiều đầu ra.

---
### `ppmi.py`

**Lớp `TruncatedSVD`**: Tương tự với lớp cùng tên trong `Tfidf.py` nhưng có một chút khác biệt, lớp `TruncatedSVD` ở đây được định nghĩa lại cho phù hợp với PPMI khi sử dụng ma trận đầu vào là ma trận thưa để tối ưu hiệu suất.

**Lớp `PPMIEmbedder`**: Đây là lớp thực hiện xây dựng mô hình biểu diễn từ dựa trên PPMI và giảm chiều bằng TruncatedSVD. Lớp xử lý đầu vào là danh sách văn bản, đầu ra là biểu diễn từ và văn bản ở dạng vector. Các phương thức bao gồm:

- `__init__`:  
  Khởi tạo mô hình với các tham số:
  - `window_size`: kích thước cửa sổ ngữ cảnh  
  - `max_features`: số lượng từ tối đa  
  - `n_components`: số chiều giảm  
  - `min_count`: tần suất tối thiểu để chọn từ  
  - `n_jobs`: số luồng xử lý song song

- `fit(documents)`:
  - Tiền xử lý văn bản và xây dựng từ điển ánh xạ từ → chỉ số  
  - Tính trọng số IDF tương ứng  
  - Xây dựng ma trận đồng xuất hiện dạng thưa  
  - Tính toán ma trận PPMI  
  - Áp dụng TruncatedSVD để giảm chiều và chuẩn hóa vector biểu diễn từ

- `transform_docs(documents)`:
  - Chuyển đổi danh sách tài liệu sang vector biểu diễn  
  - Thực hiện bằng cách lấy trung bình có trọng số IDF của các vector từ trong văn bản

- `transform(document)`:
  - Hỗ trợ suy diễn cho một tài liệu đơn hoặc danh sách tài liệu  
  - Trả về vector biểu diễn tương ứng



## 2️⃣Word Embedding models
Gồm các file: `HellingerPCA.py`, `word2Vec.py`, `glove.py`, `FastText.py`

