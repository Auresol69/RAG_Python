import os
# Bắt buộc thư viện chạy chế độ Offline
os.environ['HF_HUB_OFFLINE'] = '1'

from loaders import LoaderFactory
from splitters import RecursiveSplitter
from sentence_transformers import SentenceTransformer

def main():
    filePath = "sample.pdf"
    print("Đang đọc file PDF...")
    loader = LoaderFactory.createLoader(filePath)
    fullText = loader.load(filePath)

    print(f"Đang cắt văn bản (Độ dài gốc: {len(fullText)} ký tự)...")
    splitter = RecursiveSplitter(chunkSize=500)
    chunks = splitter.splitText(fullText)

    print(f"Đã cắt thành {len(chunks)} đoạn (chunks).")
    print(f"--- Nội dung đoạn đầu tiên ---")
    print(chunks[1])
    print("-" * 50)

    print("Đang tải Model AI (Lần đầu sẽ mất khoảng 1-2 phút)...")

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print("Đang biến đổi văn bản thành Vector...")

    vector = model.encode(chunks[1])
    
    print(f"Xong! Đoạn văn bản đã biến thành một mảng số.")
    print(f"Kích thước vector (Dimension): {vector.shape}") 
    print(f"Dữ liệu vector (5 số đầu): {vector[:5]}")

if __name__ == "__main__":
    main()