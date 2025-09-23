import os
from dotenv import load_dotenv
from pathlib import Path

# LangChain 라이브러리 임포트
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory

# .env 파일에서 환경 변수 로드
load_dotenv()

class OboeChatbot:
    """
    '오보에' 앨범 분석 PDF를 기반으로 답변하는 RAG 챗봇 클래스
    """
    def __init__(self):
        # .env 파일에서 설정값 가져오기
        self.pdf_filename = os.getenv("PDF_FILENAME")
        
        # 현재 스크립트 파일이 위치한 디렉토리를 기준으로 PDF 파일 경로 설정
        base_dir = Path(__file__).parent
        self.pdf_path = base_dir / self.pdf_filename
        
        # 벡터 스토어 및 대화 체인 초기화
        self.vector_store = self._get_or_create_vector_store()
        self.chain = self._setup_conversational_chain()

    def _get_or_create_vector_store(self):
        """PDF 파일로부터 벡터 스토어를 생성합니다."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"'{self.pdf_filename}' 파일을 찾을 수 없습니다. "
                f"backend 폴더에 파일이 있는지 확인해주세요."
            )
        
        try:
            loader = PyPDFLoader(str(self.pdf_path))
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            split_docs = text_splitter.split_documents(documents)
            
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
            
            print("✅ PDF 문서 로드 및 벡터 스토어 생성 완료.")
            return vectorstore
        except Exception as e:
            print(f"🚨 벡터 스토어 생성 중 오류 발생: {e}")
            raise

    def _setup_conversational_chain(self):
        """LangChain을 사용하여 대화형 검색 체인을 설정합니다."""
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, streaming=True)
        
        # _setup_conversational_chain 메서드 내부
        prompt_template = """
        당신은 사용자와 1:1로 대화하는 친절하고 매력적인 '음악 큐레이터'입니다. 
        당신의 이름은 '오보'이며, 양홍원의 '오보에' 앨범에 대해 모르는 것이 없는 전문가입니다. 
        딱딱한 정보 전달이 아닌, 상대방의 눈높이에 맞춰 설명하고 감상을 공유하는 역할을 합니다.

        --- 페르소나 및 대화 규칙 ---
        1.  **인사와 공감**: 답변을 시작할 때, 사용자의 질문에 대해 "좋은 질문이네요!", "그 부분에 대해 궁금하셨군요." 와 같이 가볍게 공감하며 시작해주세요.
        2.  **두괄식 설명**: 항상 핵심 결론을 먼저 말하고, 그 후에 근거를 분석 문서 내용에서 찾아 2-3문장으로 부연 설명해주세요.
        3.  **비유와 예시 활용**: 어려운 음악 용어나 개념은 "마치 ~처럼요.", "예를 들면 이런 느낌이죠." 와 같이 일상적인 비유를 들어 쉽게 설명해주세요.
        4.  **정보 출처 명시 (선택적)**: 답변의 신뢰도를 위해, 문서의 어떤 내용을 참고했는지 가볍게 언급할 수 있습니다. (예: "앨범 분석 문서에서는 이 부분을 ~라고 표현하고 있어요.")
        5.  **대화 유도**: 답변 마지막에는 항상 "혹시 더 궁금한 점이 있으신가요?", "이 곡의 다른 부분에 대해서도 이야기해볼까요?" 와 같이 자연스럽게 다음 질문을 유도해주세요.
        6.  **모를 때는 솔직하게**: 분석 문서에 없는 내용에 대한 질문을 받으면, "제가 가진 분석 자료에는 해당 내용이 없네요. 죄송하지만 다른 질문을 해주시겠어요?" 라고 솔직하고 정중하게 답변하세요. 절대 추측해서 말하지 마세요.

        --- 분석 문서 내용 ---
        {context}
        --------------------

        --- 이전 대화 ---
        {chat_history}
        ----------------

        질문: {question}
        음악 큐레이터 '오보'의 답변:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "chat_history", "question"]
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={'k': 5}),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer'),
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=False
        )
        print("✅ 대화형 챗봇 체인 설정 완료.")
        return chain

    def get_response(self, user_query: str) -> str:
        """사용자 질문에 대한 답변을 생성합니다."""
        try:
            result = self.chain.invoke({"question": user_query})
            return result.get('answer', "답변을 가져오는 데 실패했습니다.")
        except Exception as e:
            print(f"🚨 답변 생성 중 오류 발생: {e}")
            return "죄송합니다, 답변 생성 중 오류가 발생했습니다. API 키나 서버 상태를 확인해주세요."

# 전역 변수로 챗봇 인스턴스 생성
try:
    chatbot_instance = OboeChatbot()
except Exception as e:
    print(f"🚨 챗봇 초기화 실패: {e}")
    chatbot_instance = None