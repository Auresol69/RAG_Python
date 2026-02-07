import os
from abc import ABC, abstractmethod
from pypdf import PdfReader

class DocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> str:
        """Đọc file và trả về nội dung dưới dạng text"""
        pass

class PDFLoader(DocumentLoader):
    def load(self, file_path: str) -> str:
        text = ""
        try:
            reader = PdfReader(file_path)

            for page in reader.pages:
                text += page.extract_text() + "\n"
            print(f"[PDFLoader] Đã đọc xong file: {file_path}")
            return text
        except Exception as e:
            print(f"[PDFLoader] Lỗi đọc file: {e}")
            return ""
        
class LoaderFactory:
    @staticmethod
    def createLoader(file_path: str) -> DocumentLoader:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return PDFLoader()
        else:
            raise ValueError(f"Chưa hỗ trợ định dạng file: {ext}")