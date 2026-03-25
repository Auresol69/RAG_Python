from rag_core.loaders import TableLoader
from rag_core.vector_store import VectorDBClient
from rag_core.generator import GeminiAdapter
from google import genai
import os
from dotenv import load_dotenv
import requests
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

load_dotenv()

class GoldRAGManager:
    def __init__(self, data_dir="/app/datas"):
        self.llm = GeminiAdapter()
        self.db_client = VectorDBClient()
        self.loader = TableLoader(base_dir=data_dir)
        self.data_dir = data_dir

    def ingest_gold_data(self, start_date, end_date, gold_type, chat_id):
        print(f"--- Bắt đầu nạp dữ liệu vàng {gold_type} ---")
        callback_url = "http://n8n:5678/webhook/crawl-finished"

        # Load data từ CSV files theo date range
        weekly_docs = self.loader.load_by_date_range(
            start_date=start_date,
            end_date=end_date,
            gold_type=gold_type,
            data_dir=self.data_dir
        )

        if not weekly_docs:
            print("Không có dữ liệu để nạp.")
            requests.post(callback_url, json={
                "chat_id": chat_id,
                "status": "fail",
                "message": f"Không có dữ liệu để nạp ní ơi."
                },
            timeout=10)
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
        try:
            success_payload = {
                "chat_id": chat_id,
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
                "chat_id": chat_id,
                "status": "error",
                "message": str(e)
            }
            try:
                requests.post(callback_url, json=error_payload, timeout=10)
                logging.info(f"Successfully sent error callback to {callback_url}")
            except requests.RequestException as e:
                logging.error(f"Failed to send error callback: {e}")

    # results['metadatas'][0]
    def merge_df(self, metas):
        """Gộp lại toàn bộ CSV trong khoảng ngày min/max lấy từ metadatas."""
        if not metas:
            logging.error("Danh sách metadata rỗng, không thể merge DataFrame")
            return None

        gold_type = metas[0]["gold_type"]
        dates = [m["date"] for m in metas]
        start_date = min(dates)
        end_date = max(dates)

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        all_dfs = []
        for date in date_range:
            year = date.strftime('%Y')
            month = date.strftime('%m')
            day = date.strftime('%d')
            file_path = os.path.join(self.data_dir, gold_type, year, month, f"{day}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Tạo cột Ngày dùng cho biểu đồ
                df['Ngày'] = pd.to_datetime(date.date())
                all_dfs.append(df)

        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            return final_df
        else:
            logging.error("Không tìm thấy file nào trong khoảng thời gian này!")
            return None

    def draw_chart(self, metas):
        """Vẽ biểu đồ trung bình ngày và trả về ảnh dạng base64 (PNG)."""
        df = self.merge_df(metas=metas)
        if df is None or df.empty:
            logging.error("Không có dữ liệu để vẽ biểu đồ")
            return None

        if 'Ngày' not in df.columns:
            logging.error("Không tìm thấy cột 'Ngày' trong DataFrame")
            return None

        df['Ngày'] = pd.to_datetime(df['Ngày'])

        # gom và tính trung bình theo ngày
        daily_avg = df.groupby('Ngày').agg({
            'Mua vào': 'mean',
            'Bán ra': 'mean'
        }).reset_index()

        if daily_avg.empty:
            logging.error("Data sau khi group theo ngày rỗng, không vẽ được")
            return None

        plt.figure(figsize=(12, 6))

        plt.plot(daily_avg['Ngày'], daily_avg['Bán ra'], color='#e74c3c', label='Giá bán ra', linewidth=2)
        plt.plot(daily_avg['Ngày'], daily_avg['Mua vào'], color='#99f82b', label='Giá mua vào', linewidth=2)

        plt.fill_between(daily_avg['Ngày'], daily_avg['Mua vào'], daily_avg['Bán ra'], color="#7B7C79", alpha=0.2, label='Chênh lệch (Spread)')

        plt.title('Xu hướng giá vàng SJC - Phân tích biến động hàng ngày', fontsize=14)
        plt.xlabel('Thời gian')
        plt.ylabel('Giá (VNĐ)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        return img_base64

    def ask(self, question, gold_type, chat_id):
        query_vector = self.llm.embed_content(
            model="nomic-embed-text",
            contents=question,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        callback_url = "http://n8n:5678/webhook/crawl-finished"

        if not query_vector:
            requests.post(callback_url, json={"chat_id": chat_id,"status": "error", "message":"Ní ơi, lỗi tạo vector rồi, không tìm dữ liệu được!"}) 
            return "Ní ơi, lỗi tạo vector rồi, không tìm dữ liệu được!"

        results = self.db_client.search(
            collection_name=f"gold_{gold_type}_collection",
            query_vector=query_vector,
            n_results=5
        )
        logging.info(f"DEBUG - Các ngày tìm thấy: {results['metadatas'][0]}")

        if not results['documents'][0]:
            requests.post(callback_url, json={"chat_id": chat_id, "status": "error", "message":"Tui không thấy data này trong kho ní ơi."}) 
            return "Tui không thấy data này trong kho ní ơi."
        
        # Vẽ biểu đồ dựa trên khoảng ngày của các metadata tìm được
        chart_base64 = None
        try:
            metas = results['metadatas'][0]
            chart_base64 = self.draw_chart(metas=metas)
        except Exception as e:
            logging.error(f"Lỗi khi vẽ biểu đồ: {e}")

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

        payload = {
            "chat_id": chat_id,
            "status": "success",
            "message": f"{answer}"
        }

        # Nếu có biểu đồ, gửi kèm base64 về cho phía nhận xử lý hiển thị
        if chart_base64:
            payload["chart_base64"] = chart_base64
            payload["chart_format"] = "png"

        requests.post(callback_url, json=payload)
    
if __name__ == "__main__":
    # manager = GoldRAGManager(data_dir="./app/datas")
    # manager.ingest_gold_data("2014-04-01", "2014-04-30", "sjc")
    # answer = manager.ask("Giá vàng của năm 2014 như nào", "sjc")
    # print(answer)
    pass