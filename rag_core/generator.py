import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig  # Import trực tiếp để có autocomplete
from dotenv import load_dotenv

load_dotenv()

class GeminiAdapter:
    def __init__(self, model_name="gemini-flash-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Chưa tìm thấy GOOGLE_API_KEY trong file .env")
        genai.configure(api_key=api_key)

        config = GenerationConfig(
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            max_output_tokens=8192
        )
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=config
        )

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
        Bạn là trợ lý AI. Dựa vào thông tin sau để trả lời câu hỏi. 
        Nếu không có thông tin, hãy nói "Tôi không biết".
        
        📖 THÔNG TIN:
        {context}
        
        ❓ CÂU HỎI: {question}
        """
        try:
            response=self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Lỗi Gemini: {e}"