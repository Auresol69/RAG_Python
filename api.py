from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_craw import GoldRAGManager
import os
import threading
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gold RAG API", version="1.0.0")

# Khởi tạo manager
manager = GoldRAGManager(data_dir=os.getenv("DATA_DIR", "./datas"))

# ============== Models ==============
class IngestRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    gold_type: str   # sjc, ưu ái, nhẫn...
    chat_id: str

class IngestResponse(BaseModel):
    status: str
    message: str

class AskRequest(BaseModel):
    question: str
    gold_type: str  # sjc, ưu ái, nhẫn...
    chat_id: str

class AskResponse(BaseModel):
    status: str
    message: str

# ============== Endpoints ==============

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Gold RAG API - Version 1.0.0"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(request: IngestRequest):
    """
    Nạp dữ liệu vàng vào ChromaDB
    
    Example request:
    {
        "start_date": "2014-04-01",
        "end_date": "2014-04-30",
        "gold_type": "sjc"
    }
    """
    try:
        print(f"[API] Bắt đầu nạp dữ liệu từ {request.start_date} đến {request.end_date}")
        thread = threading.Thread(
            target=manager.ingest_gold_data,
            kwargs={
                "start_date": request.start_date,
                "end_date": request.end_date,
                "gold_type": request.gold_type,
                "chat_id": request.chat_id
            },
            daemon=True,
        )
        thread.start()
        return IngestResponse(
            status="accepted",
            message=f"Đang nạp dữ liệu vàng {request.gold_type} từ {request.start_date} đến {request.end_date}"
        )
    except Exception as e:
        print(f"[API] Lỗi nạp dữ liệu: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi nạp dữ liệu: {str(e)}")

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """
    Hỏi câu hỏi về dữ liệu vàng
    
    Example request:
    {
        "question": "Giá vàng của năm 2014 như nào?",
        "gold_type": "sjc"
    }
    """
    try:
        print(f"[API] Trả lời câu hỏi: {request.question}")
        thread = threading.Thread(
            target=manager.ask,
            kwargs={
                "question": request.question,
                "gold_type": request.gold_type,
                "chat_id": request.chat_id
            },
            daemon=True,
        )
        thread.start()
        return AskResponse(status="accepted", message=f"Đang phân tích giá vàng {request.gold_type}")
    except Exception as e:
        print(f"[API] Lỗi trả lời câu hỏi: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi trả lời câu hỏi: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
