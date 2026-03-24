from rag_core.loaders import TableLoader
from rag_core.vector_store import VectorDBClient
from rag_core.generator import GeminiAdapter
from google import genai
import os
from dotenv import load_dotenv
import requests
import logging

load_dotenv()

class GoldRAGManager:
    def __init__(self, data_dir="app/datas"):
        self.llm = GeminiAdapter()
        self.db_client = VectorDBClient()
        self.loader = TableLoader(base_dir=data_dir)
        self.data_dir = data_dir

    def ingest_gold_data(self, start_date, end_date, gold_type):
        print(f"--- Bắt đầu nạp dữ liệu vàng {gold_type} ---")

        # Load data từ CSV files theo date range
        weekly_docs = self.loader.load_by_date_range(
            start_date=start_date,
            end_date=end_date,
            gold_type=gold_type,
            data_dir=self.data_dir
        )

        if not weekly_docs:
            print("Không có dữ liệu để nạp.")
            return
        
        # Chia nhỏ để nạp (Batching) - Mỗi đợt 10 ngày (gần 1 năm dữ liệu)
        batch_size = 20
        for i in range(0, len(weekly_docs), batch_size):
            batch = weekly_docs[i:i+batch_size]
            # Chuẩn bị dữ liệu
            texts = [doc["content"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            # sjc_20140403
            ids = [doc["metadata"]["source"] for doc in batch]
            print(ids)
            # Embedding
            vectors = self.llm.embed_content(
                model="nomic-embed-text",
                contents=texts,
                config={'task_type': 'RETRIEVAL_DOCUMENT'}
            )

            if vectors:
                self.db_client.add_documents(
                    collection_name=f"gold_{gold_type}_collection",
                    texts=texts,
                    vectors=vectors,
                    metadatas=metadatas,
                    ids=ids
                )
            logging.info(f"Đã nạp xong {len(texts)} ngày dữ liệu vàng {gold_type} vào ChromaDB!")
        callback_url = "http://n8n:5678/webhook/crawl-finished"
        try:
            success_payload = {
                "intent": "NAP_DATA",
                "status": "success",
                "message": f"Đã nạp xong vàng {gold_type} từ {start_date} đến {end_date}"
                }
            try:
                requests.post(callback_url, json=success_payload, timeout=10)
                logging.info(f"Successfully sent success callback to {callback_url}")
            except requests.RequestException as e:
                logging.error(f"Failed to send success callback {e}")              

        except Exception as e:
            error_payload = {
                "status": "error",
                "message": str(e)
            }
            try:
                requests.post(callback_url, json=error_payload, timeout=10)
                logging.info(f"Successfully sent error callback to {callback_url}")
            except requests.RequestException as e:
                logging.error(f"Failed to send error callback: {e}")

    def ask(self, question, gold_type):
        query_vector = self.llm.embed_content(
            model="nomic-embed-text",
            contents=question,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        callback_url = "http://n8n:5678/webhook/crawl-finished"

        if not query_vector:
            requests.post(callback_url, {"status": "error", "message":"Ní ơi, lỗi tạo vector rồi, không tìm dữ liệu được!"}) 
            return "Ní ơi, lỗi tạo vector rồi, không tìm dữ liệu được!"

        results = self.db_client.search(
            collection_name=f"gold_{gold_type}_collection",
            query_vector=query_vector,
            n_results=5
        )
        logging.info(f"DEBUG - Các ngày tìm thấy: {results['metadatas'][0]}")

        if not results['documents'][0]:
            requests.post(callback_url, {"status": "error", "message":"Tui không thấy data này trong kho ní ơi."}) 
            return "Tui không thấy data này trong kho ní ơi."
        
        # matplotlib

        context = "\n\n".join(results['documents'][0])
        
        prompt = f"""
        Bạn là một chuyên gia phân tích thị trường vàng tại Việt Nam. 
        Dựa vào dữ liệu lịch sử dưới đây, hãy trả lời câu hỏi của người dùng một cách ngắn gọn, chuyên nghiệp.

        DỮ LIỆU LỊCH SỬ:
        {context}

        CÂU HỎI: "{question}"

        LƯU Ý: Nếu dữ liệu không có thông tin chính xác, hãy nói bạn không biết, đừng đoán bừa.
        """

        answer = self.llm.generate_answer(prompt=prompt)


        requests.post(callback_url, {"intent": "PHAN_TICH","status": "success", "message": f"{answer}"})
    
if __name__ == "__main__":
    # manager = GoldRAGManager(data_dir="./app/datas")
    # manager.ingest_gold_data("2014-04-01", "2014-04-30", "sjc")
    # answer = manager.ask("Giá vàng của năm 2014 như nào", "sjc")
    # print(answer)
    pass