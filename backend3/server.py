# backend3/server.py
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

# LangGraph 그래프(app) 가져오기
from my_agent.agent import app as graph_app

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

api = FastAPI(title="Oboe RAG API")

# 필요 시 CORS(동일 오리진이면 없어도 됨)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskInput(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = []

@api.get("/", response_class=HTMLResponse)
def index():
    # templates/index1.html을 그대로 반환
    return FileResponse(str(TEMPLATE_DIR / "index1.html"))

@api.get("/health")
def health():
    return {"ok": True}

@api.post("/ask")
def ask(payload: AskInput):
    # LangGraph 호출
    result: Dict[str, Any] = graph_app.invoke(
        {"question": payload.question, "chat_history": payload.chat_history}
    )
    return {
        "answer": result.get("answer", ""),
        "chat_history": result.get("chat_history", []),
        "citations": result.get("citations", []),
        "error": result.get("error", ""),
    }
