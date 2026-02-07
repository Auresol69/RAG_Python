import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("📋 DANH SÁCH MODEL CỦA BẠN:")
for m in genai.list_models():
    # Chỉ in ra các model hỗ trợ tạo nội dung (generateContent)
    if 'generateContent' in m.supported_generation_methods:
        print(f"- Name: {m.name}")