# kích hoạt môi trường ảo venv trước
source .venv/bin/activate
# Thoát môi trường ảo
deactivate
# cài đặt nhiều thư viện cùng lúc
pip install -r requirements.txt
# chạy file python
python test.py
# Kiểm tra dung lượng của riêng 1 File
ls -lh tên_file
du -schL *