# check_models.py
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

print(f"--- ĐANG KIỂM TRA VỚI KEY: {api_key[:5]}...{api_key[-5:]} ---")

try:
    print("Danh sách các model bạn có thể dùng:")
    for m in client.models.list():
        # In ra tất cả tên model để mình copy-paste cho chính xác
        print(f"- {m.name}")
except Exception as e:
    print(f"❌ Lỗi khi lấy danh sách: {e}")