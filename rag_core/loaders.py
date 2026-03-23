import os
from abc import ABC, abstractmethod
from pypdf import PdfReader
import pandas as pd

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

class TableLoader(DocumentLoader):
    """Load Time Series data từ CSV files"""
    def __init__(self, base_dir: str = "./app/datas"):
        self.base_dir = base_dir
    
    def load(self, file_path: str) -> str:
        """Đọc CSV file và trả về nội dung dưới dạng text"""
        try:
            if not os.path.exists(file_path):
                return ""
            
            df = pd.read_csv(file_path)
            return self._df_to_text(df)
        
        except Exception as e:
            print(f"[TableLoader] Lỗi: {e}")
            return ""
    
    def load_by_date_range(self, start_date: str, end_date: str, gold_type: str, data_dir: str = None) -> list:
        """Load data trong khoảng thời gian từ multiple CSV files
        
        Args:
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            data_dir: Thư mục chứa data (default: self.base_dir)
            
        Returns:
            List of {"content": str, "metadata": {...}}
        """
        if data_dir is None:
            data_dir = self.base_dir
            
        documents = []
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        for d in date_range:
            year, month, day = d.strftime('%Y'), d.strftime('%m'), d.strftime('%d')
            file_path = os.path.join(data_dir, gold_type, year, month, f"{day}.csv")

            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path).dropna(how='all')
                    if df.empty: continue
                
                    # Biến dữ liệu 1 ngày thành text
                    day_content = self._df_to_text(df)
                
                    # Tạo metadata và ID riêng cho ngày đó
                    metadata = {
                        "date": str(d.date()),
                        "gold_type": gold_type,
                        "source": f"{gold_type}_{d.strftime('%Y%m%d')}" # ID duy nhất: sjc_20140401
                    }

                    documents.append({"content": day_content, "metadata": metadata})
                except Exception as e:
                    print(f"Lỗi đọc file {file_path}: {e}")
        
        # Gộp tất cả thành 1 chuỗi text
        if not documents:
            return []
        
        return documents

    def _df_to_text(self, df: pd.DataFrame) -> str:
        """Helper biến các hàng dữ liệu thành câu văn cho AI dễ hiểu"""
        # Dùng apply nhanh hơn iterrows rất nhiều
        lines = df.apply(
            lambda row: f"Ngày {row['Thời gian']}, vàng {row['Loại vàng']} tại {row['Khu vực']} có giá mua vào {row['Mua vào']} và bán ra {row['Bán ra']}.", 
            axis=1
        )
        return "\n".join(lines.tolist())

class LoaderFactory:
    @staticmethod
    def createLoader(file_path: str) -> DocumentLoader:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return PDFLoader()
        elif ext == '.csv':
            return TableLoader()
        else:
            raise ValueError(f"Chưa hỗ trợ định dạng file: {ext}")