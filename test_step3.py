import os
os.environ['HF_HUB_OFFLINE'] = '1'

from rag_core.splitters import RecursiveSplitter
from rag_core.loaders import LoaderFactory
from sentence_transformers import SentenceTransformer
from rag_core.vector_store import VectorDBClient
import uuid
def main():
    file_path = "sample.pdf"
    loader = LoaderFactory.createLoader(file_path)
    text = loader.load(file_path)

    splitter = RecursiveSplitter(chunkSize=500)
    chunks = splitter.splitText(text=text)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    vectors = model.encode(chunks)

    db_client = VectorDBClient()

    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    metadatas = [{"source": file_path} for _ in range(len(chunks))]

    db_client.add_documents(
        collection_name="demo_rag",
        ids= ids,
        metadatas= metadatas,
        texts=chunks,
        vectors=vectors.tolist() # Chroma yêu cầu list, không phải numpy array
    )

    # Câu hỏi trực tiếp: "Ai là Giám đốc kỹ thuật của dự án Neura-Core?"

    # Câu hỏi về số liệu: "Ngân sách dành cho giai đoạn huấn luyện mô hình là bao nhiêu?"

    # Câu hỏi yêu cầu tổng hợp: "Hệ thống này sử dụng những công nghệ gì để giảm thiểu sai sót thông tin?"

    # Câu hỏi so sánh: "So với phiên bản V1, phiên bản này có ưu điểm gì về năng lượng?"


    db_client_2 = VectorDBClient()
    question = "Ai là Giám đốc kỹ thuật của dự án Neura-Core?"

    query_vector = model.encode(question).tolist()
    results = db_client_2.search(
        collection_name="demo_rag",
        n_results=1,
        query_vector= query_vector
    )
    print(results['documents'][0][0])



if __name__ == "__main__":
    main()