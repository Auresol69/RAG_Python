import os
from ollama import Client
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GeminiAdapter:
    def __init__(self, model_name="gemini-flash-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Chưa tìm thấy GOOGLE_API_KEY trong file .env")
        self.gemini_client = genai.Client(
            api_key=api_key,)

        self.generation_config = {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "max_output_tokens": 8192
        }
        self.ollama_client = Client(host='http://ollama:11434')
        self.model_name = model_name

    def embed_content(self, model, contents, config):
        """Dùng Ollama trong Docker để tạo vector"""
        try:
            # 1. Đảm bảo model đã được tải TRONG DOCKER
            # cần run: docker exec -it ollama ollama pull nomic-embed-text
            options = {'num_ctx': 8192}

            if isinstance(contents, list):
                embeddings = []
                for text in contents:
                    res = self.ollama_client.embeddings(
                        model="nomic-embed-text", 
                        prompt=text,
                        options=options)
                    embeddings.append(res['embedding'])
                return embeddings
            
            res = self.ollama_client.embeddings(
                model="nomic-embed-text", 
                prompt=contents,
                options=options   
            )
            return res['embedding']
            
        except Exception as e:
            print(f"Lỗi kết nối tới Docker Ollama: {e}")
            return []

    def generate_answer(self, prompt: str) -> str:
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config
            )
            return response.text
        except Exception as e:
            return f"Lỗi AI: {e}"
        

class GeminiDataAgent:
    def __init__(self, model_name="gemini-3-flash-preview"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: raise ValueError("Thiếu API KEY")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        # CẤU HÌNH CÔNG CỤ: Cho phép Gemini tự viết và chạy code Python
        self.tools_config = [
            types.Tool(code_execution=types.ToolCodeExecution())
        ]

    def analyze_and_plot(self, prompt: str, file_paths: list):
        """Flow Đặc vụ: Vừa phân tích, vừa viết code vẽ biểu đồ"""
        try:
            # Gửi prompt kèm theo danh sách file CSV (Gemini cần biết đường dẫn file)
            # Bạn có thể tải file lên Gemini File API nếu file quá lớn
            
            enhanced_prompt = f"""
            Bạn là chuyên gia phân tích dữ liệu vàng. 
            Dựa trên danh sách các file CSV dữ liệu giá vàng năm 2014 dưới đây.
            Hãy viết code Python để:
            1. Gom tất cả dữ liệu từ các file này thành một Pandas DataFrame.
            2. Tính giá trung bình mua/bán theo từng tháng.
            3. Vẽ biểu đồ đường so sánh giá mua/bán trung bình cả năm.
            4. Trả về nhận xét về xu hướng giá.

            DANH SÁCH FILE:
            {file_paths}
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=enhanced_prompt,
                config=types.GenerateContentConfig(tools=self.tools_config) # Kích hoạt tool
            )
            
            # Xử lý kết quả đa năng (Multi-part response)
            final_text = ""
            generated_code = ""
            
            for part in response.candidates[0].content.parts:
                if part.text:
                    final_text += part.text
                if part.executable_code:
                    generated_code = part.executable_code.code # Gemini trả về code nó đã viết
            
            return final_text, generated_code

        except Exception as e:
            return f"❌ Lỗi Đặc vụ: {e}", None