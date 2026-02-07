import threading, chromadb
class VectorDBClient:
    """giống static instance"""
    _instance = None
    """giống synchronized JAVA"""
    _lock = threading.Lock()
    client: chromadb.ClientAPI  # Type hint để có autocomplete
    """Tạo vỏ rỗng"""
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(VectorDBClient, cls).__new__(cls)
                    cls._instance.client = chromadb.PersistentClient(path="chroma_db")
        return cls._instance
    
    def get_collection(self, name="rag_collection"):
        """Lấy hoặc tạo mới một bảng (collection) trong DB"""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, collection_name, texts, vectors, metadatas=None, ids=None):
        """Lưu văn bản và vector vào kho"""
        collection = self.get_collection(collection_name)

        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        collection.add(
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Đã lưu {len(texts)} documents vào collection '{collection_name}'")

    def search(self, collection_name,query_vector, n_results=2):
        collection = self.get_collection(collection_name)

        results = collection.query(
            # [query_vector]: Là danh sách chứa nhiều câu hỏi.
            query_embeddings=[query_vector],
            n_results=n_results
        )

        return results