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
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4, streaming=True)
        
        prompt_template = """
        당신은 양홍원의 '오보에' 앨범 분석 문서를 완벽히 숙지한 전문 AI 음악 평론가입니다.
        주어진 '분석 문서 내용'을 바탕으로, 사용자의 '질문'에 대해 깊이 있고 통찰력 있는 답변을 제공해주세요.
        답변은 반드시 문서 내용에 근거해야 하며, 없는 내용은 절대 추측하지 마세요.
        친절하고 전문적인 말투로 답변해주세요.

        --- 분석 문서 내용 ---
        {context}
        --------------------

        --- 이전 대화 ---
        {chat_history}
        ----------------

        질문: {question}
        전문가 답변:
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