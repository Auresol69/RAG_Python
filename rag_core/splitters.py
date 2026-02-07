from abc import ABC, abstractmethod

class TextSplitter(ABC):
    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Cắt văn bản dài thành các đoạn nhỏ (chunks)"""
        pass

class CharacterSplitter(TextSplitter):
    """Chiến thuật cắt đơn giản: Cắt theo số lượng ký tự cố định"""
    def __init__(self, chunk_size=1000, overlap=100):
        self.chunk_size = chunk_size
        """
        Overlap (Phần gối đầu) hiểu đơn giản là: Khi bạn cắt văn bản thành từng khúc,
        khúc sau sẽ lặp lại một đoạn cuối của khúc trước.
        
        Văn bản gốc:  [ A B C D E F G H I J ]
        Chunk 1:      [ A B C D E F ]
        Chunk 2:              [ E F G H I J ]
                              <---> 
                        Khu vực Overlap
        """
        self.overlap = overlap
    
    def split_text(self, text) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks
    
class RecursiveSplitter(TextSplitter):
    """Chiến thuật cắt thông minh (Simplified): Ưu tiên cắt theo dấu xuống dòng"""
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size

    def split_text(self, text) -> list[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        # para cuối
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks