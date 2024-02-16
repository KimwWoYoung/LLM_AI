from tqdm import tqdm
import time
import os
import json
import pickle
import regex as re
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import pinecone

# 필요한 전역 변수 선언
global embedder, pc, pc_index, vectorstore, llm, qa_chain, case_map

def initialize_system():
    global embedder, pc, pc_index, vectorstore, llm, qa_chain, case_map

    # 환경변수 설정 및 API 키 로드
    with open('apikeys.json', 'r') as f:
        API_KEYS = json.load(f)

    os.environ['PINECONE_API_KEY'] = API_KEYS['pinecone']['key']
    os.environ['PINECONE_API_ENV'] = API_KEYS['pinecone']['env']
    OPENAI_API_KEY = API_KEYS['open_ai']['key']

    PINECONE_INDEX_NAME = 'llm-rag-openai' # 파인콘 인덱스 이름
    EMBEDDER_NAME = 'text-embedding-ada-002'
    LLM_NAME = 'gpt-3.5-turbo-0125'

    with tqdm(total=100, leave=True) as pbar:
        pbar.set_description("임베더 불러오는중..")
        embedder = OpenAIEmbeddings(model=EMBEDDER_NAME, openai_api_key=OPENAI_API_KEY)
        pbar.update(25)

        pbar.set_description("파인콘 연결중..")
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_API_ENV'))
        for _ in range(5):
            if pc.describe_index(PINECONE_INDEX_NAME).status.ready is not True:
                time.sleep(1)
            else:
                pc_index = pc.Index(PINECONE_INDEX_NAME)
                break
        else:
            raise RuntimeError("파인콘 오류 : Index 연결불가")
        pbar.update(25)

        pbar.set_description("사건번호 맵 불러오기..")
        with open('case_map.pkl', 'rb') as file:
            case_map = pickle.load(file)
        pbar.update(25)

        pbar.set_description("파인콘&랭체인 연동중..")
        vectorstore = pinecone.Pinecone(index=pc_index, embedding=embedder, text_key='text')
        pbar.update(25)

        pbar.set_description("LLM 로딩중..")
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.1, max_tokens=1024, model_name=LLM_NAME)
        qa_chain = load_qa_chain(llm, chain_type='stuff')
        pbar.update(25)

        print("#######준비완료#######")
        print(f"임베딩모델 : {EMBEDDER_NAME}\nLLM : {LLM_NAME}")

def llm_gen(input_text:str):
    """llm 텍스트 생성 함수"""
    global case_map
    # 설명, 요약, 정리 키워드와 판례번호가 패턴에 존재하면 필터링 서치, 아니면 semantic search
    match1 = re.search(r"설명|요약|정리", input_text)
    input_case_num = case_num_parser(input_text, case_map)
    if match1 and input_case_num:
        retrieved_chunks = vectorstore.similarity_search(input_text, k=10, filter={"case_num":input_case_num})
    else:
        retrieved_chunks = vectorstore.similarity_search(input_text, k=4)
    case_nums = {i.metadata['case_num'] for i in retrieved_chunks}
    
    output_text = qa_chain.run(input_documents=retrieved_chunks, question=input_text)
    return output_text, case_nums

def case_num_parser(text, case_map:dict):
    """입력 텍스트에서 판례번호를 찾고 맞는 형태로 파싱해주는 함수"""
    match = re.search(r"\d+\s*[가-힣]+\s*\d+", text)
    if match:
        key = re.search(r"\d{3,}$", match.group())
    elif (match2:=re.search(r"\d{3,}", text)):
        key = match2
    else:
        key = None
    
    if key and key.group() in case_map:
        ans = case_map[key.group()]
    else:
        ans = ''
    return ans


# 판례 상세 정보를 검색하는 함수 예시
def get_case_detail(case_num: str):
    return f"{case_num} 판례에 대한 상세 정보입니다. [판례 상세 설명]"


# 사용자 요청에 따라 판례 상세 정보를 제공하거나 일반 질문에 답변하는 함수
def bot_respond(input_text):
    global chat_hist
    # 수정된 반환 값 처리
    case_nums = None  # 초기화
    if "대법" in input_text and len(input_text.split()) == 1:
        output_text = get_case_detail(input_text)
        case_nums = [input_text]  # 예시로 판례 번호를 case_nums에 추가
    else:
        output_text, case_nums = llm_gen(input_text)
        output_text += '\n\n참고판례 : ' + ", ".join(case_nums)
        chat_hist.append((input_text, output_text))
    return output_text, case_nums

