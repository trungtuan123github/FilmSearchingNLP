# 📃Cấu trúc folder

```
preprocessing/
|-- __init__.py
|-- mongodb_connector.py       # File chứa các phương thức để kết nối tới MongoDB để lấy dữ liệu
|-- qdrant_connector.py		    # File chứa các phương thức để kết nối và điều chỉnh với Qdrant
|-- README.md             
```

# 💭Mô tả
Folder chứa các phương thức để liên kết đến database. Có hai database: 
- **MongoDB** để lưu trữ dữ liệu phim.
- **Qdrant** để lưu trữ vector số được nhúng từ văn bản.
