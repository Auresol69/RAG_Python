# Dùng bản 3.12 cho đúng với môi trường Bao vừa cài thành công
FROM python:3.12-slim

# Cài đặt các thư viện hệ thống cần thiết cho ChromaDB và Pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy file requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào
COPY . .

# Chạy app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]