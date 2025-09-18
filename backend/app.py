from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chatbot_instance

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def ask_question():
    if chatbot_instance is None:
        return jsonify({'error': '챗봇이 초기화되지 않았습니다. 서버 로그를 확인해주세요.'}), 500

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': '질문이 필요합니다.'}), 400
    
    question = data['question']
    
    response = chatbot_instance.get_response(question)
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    if chatbot_instance:
        print("🚀 Flask 서버가 시작됩니다. http://127.0.0.1:5000 에서 실행 중입니다.")
        app.run(host='0.0.0.0', port=5000)
    else:
        print("❌ 챗봇 인스턴스가 없어 서버를 시작할 수 없습니다.")