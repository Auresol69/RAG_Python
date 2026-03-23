import os
os.environ['HF_HUB_OFFLINE'] = '1' # Chạy offline

from rag_core.loaders import LoaderFactory
from rag_core.splitters import RecursiveSplitter
from rag_core.vector_store import VectorDBClient
from rag_core.generator import GeminiAdapter
from sentence_transformers import SentenceTransformer
import uuid

class RAGSystemFacade:
    def __init__(self):
        print("Khởi động RAG System...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.db_client = VectorDBClient()
        self.llm = GeminiAdapter()
        self.splitter = RecursiveSplitter()

    def ingest_data(self, file_path):
        print(f"Đang nạp file: {file_path}")
        loader = LoaderFactory().createLoader(file_path=file_path)
        text = loader.load(file_path=file_path)

        chunks = self.splitter.split_text(text=text)
        vectors = self.embedding_model.encode(chunks)

        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas= [{"source": file_path} for _ in range(len(chunks))]
 
        # Khi gọi add_documents, giống như đang xếp một cuốn sách vào thư viện:

        # dán nhãn ID cho nó (ID Index).

        # ghi nó vào danh sách "Sách năm 2024" (Metadata Index).

        # đặt nó vào khu vực "Sách về nấu ăn" dựa trên nội dung của nó (Vector Index)

        self.db_client.add_documents(
            vectors=vectors.tolist(),
            collection_name="demo_rag",
            ids=ids,
            metadatas=metadatas,
            texts=chunks
        )
        print("Nạp xong dữ liệu!")

    def ask(self, question):

        query_vector = self.embedding_model.encode(question)
        results = self.db_client.search(
            collection_name="demo_rag",
            n_results=1,
            query_vector=query_vector.tolist()
            )
        
        if not results['documents']:
            print("Không tìm thấy thông tin.")
            return
        
        distances = results["distances"][0] # Lấy danh sách điểm số
        SAFE_THRESHOLD = 0.5

        if not distances or distances[0] > SAFE_THRESHOLD:
            print("Thông tin này lạ quá, xin từ chối trả lời bừa.")
            return
        
        context = "\n\n".join(results['documents'][0])

        print("AI đang đọc tài liệu và thinking...")
        answer = self.llm.generate_answer(
            prompt = f"""
            Bạn là trợ lý AI. Dựa vào thông tin sau để trả lời câu hỏi. 
            Nếu không có thông tin, hãy nói "Tôi không biết".
        
            THÔNG TIN:
            {context}
        
            CÂU HỎI: {question}
            """
        )

        print("-" * 50)
        print("TRẢ LỜI:")
        print(answer)
        print("-" * 50)
        pass

if __name__ == "__main__":
    app = RAGSystemFacade()

    file_path = "sample.pdf"
    app.ingest_data(file_path=file_path)

    # app.ask("Ai là Giám đốc kỹ thuật của dự án Neura-Core?")

    #app.ask("Ngân sách dành cho giai đoạn huấn luyện mô hình là bao nhiêu?")

    app.ask("Ông Donald Trump là ai?")