# backend2/server.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# .env 로드 (OPENAI_API_KEY, OBOE_PDF_PATH, CHROMA_DIR, ALLOW_WEB 등)
load_dotenv()

# LangGraph 그래프(app) 불러오기
# backend2 폴더 안에서 `python server.py`로 실행하는 전제를 둡니다.
from app.graph import app as graph_app  # validate → retrieve → generate → finalize

server = Flask(__name__)
# 개발 중 편의를 위해 /ask CORS 허용(배포 시에는 origin을 제한하십시오)
CORS(server, resources={r"/ask": {"origins": "*"}})


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return False


@server.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@server.post("/ask")
def ask():
    """
    요청 JSON:
      {
        "question": "…",              # 필수
        "chat_history": [ ... ],      # 선택  [{role, content}]
        "allow_web": true|false       # 선택  (미지정 시 .env의 ALLOW_WEB 사용)
      }
    응답 JSON:
      {
        "answer": "…",
        "chat_history": [ ... ],
        "citations": [ ... ],
        "error": ""
      }
    """
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    question: str = str(data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question 필드가 비었습니다."}), 400

    chat_history: List[Dict[str, str]] = data.get("chat_history") or []
    allow_web = data.get("allow_web")
    if allow_web is not None:
        allow_web = _as_bool(allow_web)

    try:
        state_in = {"question": question, "chat_history": chat_history}
        if allow_web is not None:
            state_in["allow_web"] = allow_web

        result = graph_app.invoke(state_in)

        # 에러 우선 처리
        if result.get("error"):
            return jsonify({
                "answer": "",
                "chat_history": result.get("chat_history", chat_history),
                "citations": result.get("citations", []),
                "error": result["error"],
            }), 500

        return jsonify({
            "answer": result.get("answer", ""),
            "chat_history": result.get("chat_history", chat_history),
            "citations": result.get("citations", []),
            "error": "",
        }), 200

    except Exception as e:
        return jsonify({"answer": "", "chat_history": chat_history, "citations": [], "error": str(e)}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "true").strip().lower() in ("1", "true", "yes", "on")
    server.run(host=host, port=port, debug=debug)
