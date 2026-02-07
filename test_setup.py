import pypdf
print("Cài đặt pypdf thành công! Phiên bản:", pypdf.__version__)

try:
    import chromadb
    print("Cài đặt ChromaDB thành công!")
except ImportError:
    print("Lỗi cài đặt ChromaDB")