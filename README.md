### 이 프로젝트는 양홍원의 콘셉트 앨범 『오보에』의 상세 분석 문서를 기반으로 사용자의 질문에 답변하는 AI 챗봇을 제공
<p align= 'center'>
<img src ="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTMWfiUdNfejg-lUYvz1JWEIvBO2P-ult3xlQ&s"
</p>

사용자는 웹 인터페이스를 통해 앨범의 전체적인 서사, 각 트랙의 숨겨진 의미, 가사의 중의적 표현, 팬들의 해석 등 다채로운 질문을 자유롭게 할 수 있습니다.

AI가 잘못된 정보를 생성하는 '환각(Hallucination)' 현상을 방지하기 위해, **RAG (Retrieval-Augmented Generation, 검색 증강 생성)** 기술을 핵심 아키텍처로 사용했습니다. 이를 통해 챗봇은 오직 제공된 분석 문서의 내용에 근거하여 신뢰도 높고 통찰력 있는 답변을 생성합니다.

## 주요 기능

- **대화형 챗봇 인터페이스**: 사용자가 자연어로 질문하고 AI의 분석적인 답변을 실시간으로 받을 수 있습니다.
- **RAG 기반의 신뢰성 있는 답변**: 모든 답변은 사전에 학습된 `oboe.pdf` 분석 문서를 기반으로 생성됩니다.
- **깊이 있는 가사 분석**: 앨범의 유기적 서사, 핵심 메타포(가면, 나무, 아이 등), 시간의 흐름과 같은 복잡한 요소를 이해하고 설명합니다.
- **안정적인 웹 애플리케이션**:
    - **Backend**: `Flask`를 사용한 API 서버와 `LangChain`으로 구현된 RAG 파이프라인.
    - **Frontend**: `HTML`, `TailwindCSS`, `JavaScript`로 구축된 직관적이고 미려한 사용자 인터페이스.

## 기술 스택 및 작동 원리

이 프로젝트는 프론트엔드, 백엔드 API 서버, 그리고 AI 챗봇 코어로 구성되어 있습니다.

### 1. Frontend (`index.html`)

- 사용자 인터페이스를 제공하며, 사용자가 입력한 질문을 백엔드 API로 전송합니다.
- `fetch` API를 사용하여 비동기적으로 서버와 통신합니다.
- 서버로부터 받은 AI의 답변을 채팅창에 동적으로 렌더링합니다.

### 2. Backend API (`app.py`)

- `Flask` 프레임워크를 사용하여 간단한 웹 서버를 구축합니다.
- `/ask` 엔드포인트를 통해 POST 요청으로 사용자의 질문을 받습니다.
- 미리 생성된 `chatbot_instance`를 호출하여 질문에 대한 답변을 생성하고, 이를 JSON 형식으로 프론트엔드에 반환합니다.

### 3. AI Chatbot Core (`chatbot.py` - RAG Pipeline)

챗봇의 핵심 로직은 **LangChain** 라이브러리를 통해 구현된 RAG 파이프라인입니다. 이 과정은 크게 '인덱싱'과 '검색 및 생성' 두 단계로 나뉩니다.

### 1단계: 인덱싱 (서버 시작 시 1회 실행)

1. **문서 로드 (Load)**: `PyPDFLoader`가 `oboe.pdf` 파일의 텍스트를 불러옵니다.
2. **분할 (Split)**: `RecursiveCharacterTextSplitter`가 로드된 문서를 LLM이 처리하기 용이한 작은 단위(Chunk)로 분할합니다.
3. **임베딩 (Embed)**: `OpenAIEmbeddings` 모델이 각 텍스트 청크를 의미를 담은 숫자 벡터(Vector)로 변환합니다.
4. **저장 (Store)**: 변환된 벡터들은 `FAISS` 벡터 저장소에 저장되어, 빠르고 효율적인 유사도 검색이 가능한 '지식 창고'를 구축합