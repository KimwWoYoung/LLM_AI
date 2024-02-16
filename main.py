from class_model import initialize_system, llm_gen,  bot_respond
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Union
import uvicorn
import nest_asyncio

a = initialize_system()
chat_hist = []
# source Scripts/activate 활성화
#  uvicorn main:app --reload 
# Define FastAPI app

# 판례 상세 정보를 검색하는 함수 예시
def get_case_detail(case_num: str):
    return f"{case_num} 판례에 대한 상세 정보입니다. [판례 상세 설명]"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    input_text: str

class ResponseModel(BaseModel):
    output_text: str
    case_nums: Union[List[str], None] = None

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


def bot_respond_2(input_text):
    global chat_hist
    # 수정된 반환 값 처리
    case_nums = None  # 초기화
    if "대법" in input_text and len(input_text.split()) == 1:
        output_text = get_case_detail(input_text)
        #case_nums = [input_text]  # 예시로 판례 번호를 case_nums에 추가
    else:
        output_text, case_nums = llm_gen(input_text)
    return output_text


# HTML 페이지를 위한 GET 엔드포인트
@app.get("/query/", response_class=HTMLResponse)
async def get_query_page(request: Request):
    return templates.TemplateResponse("query.html", {"request": request})

# 사용자의 질문을 처리하고 결과 페이지를 반환하는 POST 엔드포인트
@app.post("/query/", response_class=HTMLResponse)
async def post_query_page(request: Request, input_text: str = Form(...)):
    output_text, case_nums = bot_respond(input_text)
    return templates.TemplateResponse("result.html", context={"request": request, "output_text": output_text, "case_nums": case_nums})


@app.get("/query/detail/{case_num}", response_class=HTMLResponse)
async def get_case_detail_page(request: Request, case_num: str):
    # 여기에서 case_num에 대한 상세 정보를 조회합니다.
    input_text = case_num + "에 대해 요약해줘"  # case_num 변수를 올바르게 사용
    output_text, case_nums = bot_respond(input_text)  # bot_respond 함수 호출 수정
    # 반환된 output_text를 case_detail로 사용
    return templates.TemplateResponse("detail.html", {"request": request, "case_num": case_num, "case_detail": output_text})

# FastAPI 실행을 위한 메인 코드
if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="localhost", port=8000, log_level="info")

