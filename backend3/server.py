# backend3/server.py
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from my_agent.agent import app as oboe_agent_app # agent.py에서 만든 LangGraph app

# Flask 앱 초기화. templates 폴더에서 index1.html을 관리합니다.
app = Flask(__name__, template_folder='templates')
CORS(app)  # 다른 주소(file://)의 HTML에서 오는 요청을 허용합니다.

# 대화 기록을 세션별로 관리하기 위한 간단한 인메모리 딕셔너리
# 실제 프로덕션에서는 Redis나 DB를 사용하는 것이 좋습니다.
chat_histories = {}

@app.route('/')
def index():
    """웹 브라우저에서 기본 주소로 접속하면 index1.html을 보여줍니다."""
    return render_template('index1.html')

@app.route('/ask', methods=['POST'])
def ask():
    """index1.html에서 질문을 받아 처리하는 API 엔드포인트입니다."""
    data = request.json
    question = data.get('question')

    # 간단하게 클라이언트의 IP 주소를 사용하여 사용자별 대화 기록을 구분합니다.
    client_id = request.remote_addr

    if not question:
        return jsonify({"error": "질문이 비어있습니다."}), 400

    # 해당 사용자의 이전 대화 기록을 가져옵니다. 없으면 빈 리스트로 시작합니다.
    history = chat_histories.get(client_id, [])

    # LangGraph Agent에 전달할 입력값
    inputs = {
        "question": question,
        "chat_history": history,
        "allow_web": True  # 웹 검색 허용 (tools.py의 기본값을 따르게 하려면 None)
    }

    try:
        # LangGraph Agent 실행
        # .stream() 대신 .invoke()를 사용하여 최종 결과만 한 번에 받습니다.
        final_state = oboe_agent_app.invoke(inputs)

        # Agent 실행 중 오류가 발생했다면, 오류 메시지를 반환합니다.
        if final_state.get("error"):
            error_message = f"오류가 발생했습니다: {final_state['error']}"
            return jsonify({"error": final_state["error"], "answer": error_message})

        # 오류가 없다면, 다음 대화를 위해 대화 기록을 업데이트합니다.
        chat_histories[client_id] = final_state.get("chat_history", [])
        
        # 생성된 답변을 HTML로 반환합니다.
        return jsonify({"answer": final_state.get("answer", "답변을 생성하지 못했습니다.")})

    except Exception as e:
        # 서버 자체에서 예외 발생 시 처리
        return jsonify({"error": str(e), "answer": f"죄송합니다, 서버 처리 중 심각한 오류가 발생했습니다: {str(e)}"}), 500

if __name__ == '__main__':
    # 서버를 127.0.0.1:5000 주소로 실행합니다.
    # index1.html의 fetch 주소와 일치합니다.
    print("🚀 OBOE AI 서버를 시작합니다. http://127.0.0.1:5000 에서 접속하세요.")
    app.run(host='127.0.0.1', port=5000, debug=True)