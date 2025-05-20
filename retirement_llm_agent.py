import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, Response
import configparser
app = FastAPI()

OLAMA_API_URL = "http://localhost:11434/api/generate"

class QuestionRequest(BaseModel):
    question: str

def query_olama(question: str) -> str:


    prompt_template = f"""
    당신은 퇴직연금 전문가 AI입니다.  
    사용자의 질문을, 특히 펀드 검색 관련 질문을 정확히 분석하여 두 가지 의도로 파악한 후 응답해주세요.

    **사용자의 질문:** "{question}"

    ### **단계 1: 질문 의도 파악**
    사용자가 진짜로 알고 싶어하는 것이 무엇인지 두 가지 관점에서 파악하세요:
    - 의도 1: 사용자의 명시적인 질문 의도 (표면적으로 드러난 질문)
    - 의도 2: 펀드 검색 관련 구체적 조건 분석 (아래 조건 중 해당하는 항목을 모두 포함)
       * 수익률 기간: 단기(3개월)/중기(1년)/장기(3년) 중 관심 있는 기간
       * 펀드 유형: 국내주식형/해외주식형/채권형/혼합형 등 원하는 펀드 종류
       * 위험 수준: 저위험/중위험/고위험 중 선호하는 위험도
       * 판단 조건 : 국내주식형/해외주식형 인지 모를때 ALL로 펀드 유형 정의 

    ### **단계 2: 카테고리 분류**
    파악한 의도를 바탕으로 아래 카테고리 중 가장 적합한 것을 선택하세요:
    ✅ [정보]: 퇴직연금 개념, 제도, 세금, DC형/DB형, IRP 계좌 등 일반적인 정보 질문
    ✅ [펀드]: 퇴직연금 펀드, 투자 방법, 상품 추천, 수익률 등 관련 질문
    ✅ [모두]: 정보와 펀드 모두에 관련된 복합적인 질문

    ### **단계 3: 종합 응답**
    아래 형식으로 정확히 출력하세요:

    **의도 분석**
    의도 1: [사용자의 명시적 질문 의도를 한 문장으로 서술]
    의도 2: [펀드 검색 조건 - 수익률 기간: 단기/중기/장기/ALL,펀드 유형: 국내주식형/해외주식형/채권형/혼합형, 위험 수준: 저위험/중위험/고위험]

    **카테고리**: [정보] 또는 [펀드] 또는 [모두] 중 하나만 선택
    """

    payload = {
        "model": "qwen2.5:3b-instruct",
        "prompt": prompt_template,
        "stream": False
    }

    response = requests.post(OLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "응답이 없습니다.")
    else:
        raise HTTPException(status_code=response.status_code, detail="Olama API 호출 실패")


@app.post("/query/")
def process_question(request: QuestionRequest):
    print("================== 1")
    answer = query_olama(request.question)
    print("================== 2")
    return {"answer": answer}


@app.get("/html", response_class=HTMLResponse)
async def get_simple_search_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>퇴직연금 전문가 API 테스트</title>
        <style>
            body {
                font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .input-section {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .output-section {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                min-height: 200px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
                min-height: 80px;
                font-family: inherit;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            .loading {
                display: none;
                margin-top: 10px;
                color: #7f8c8d;
            }
            .example-questions {
                margin-top: 30px;
                background-color: #eaf2f8;
                padding: 15px;
                border-radius: 8px;
            }
            .example-questions h3 {
                margin-top: 0;
                color: #2c3e50;
            }
            .example-category {
                margin-bottom: 15px;
            }
            .example-category h4 {
                margin-top: 10px;
                margin-bottom: 8px;
                color: #2c3e50;
                border-left: 3px solid #3498db;
                padding-left: 10px;
            }
            .example-questions ul {
                padding-left: 20px;
            }
            .example-questions li {
                margin-bottom: 8px;
                cursor: pointer;
                color: #2980b9;
            }
            .example-questions li:hover {
                text-decoration: underline;
            }
            .response-category {
                font-weight: bold;
                margin-bottom: 10px;
                color: #2c3e50;
                background-color: #e8f4fc;
                padding: 5px 10px;
                border-radius: 4px;
                display: inline-block;
            }
            #response-text {
                white-space: pre-line;
            }
            .api-status {
                display: flex;
                align-items: center;
                margin-top: 10px;
                font-size: 14px;
            }
            .status-indicator {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #e74c3c;
                margin-right: 6px;
            }
            .status-indicator.connected {
                background-color: #2ecc71;
            }
        </style>
    </head>
    <body>
        <h1>퇴직연금 전문가 API 테스트</h1>

        <div class="container">
            <div class="input-section">
                <h2>질문 입력</h2>
                <textarea id="question" placeholder="퇴직연금에 대한 질문을 입력하세요..."></textarea>
                <button id="submit-btn">질문하기</button>
                <div id="loading" class="loading">응답을 불러오는 중...</div>
                <div class="api-status">
                    <div id="status-indicator" class="status-indicator connected"></div>
                    <span id="status-text">API 서버 연결됨</span>
                </div>
            </div>

            <div class="output-section">
                <h2>AI 응답</h2>
                <div id="response-category" class="response-category"></div>
                <div id="response-text"></div>
            </div>

            <div class="example-questions">
                <h3>예시 질문</h3>
                <div class="example-category">
                    <h4>일반 정보 ([ASSISTANT] 카테고리)</h4>
                    <ul>
                        <li onclick="setQuestion('퇴직연금 제도에 대해 간략히 설명해주세요.')">퇴직연금 제도에 대해 간략히 설명해주세요.</li>
                        <li onclick="setQuestion('DC형과 DB형의 차이점은 무엇인가요?')">DC형과 DB형의 차이점은 무엇인가요?</li>
                        <li onclick="setQuestion('IRP 계좌란 무엇인가요?')">IRP 계좌란 무엇인가요?</li>
                        <li onclick="setQuestion('퇴직연금 세금 혜택에 대해 알려주세요.')">퇴직연금 세금 혜택에 대해 알려주세요.</li>
                    </ul>
                </div>

                <div class="example-category">
                    <h4>펀드/투자 정보 ([FUND] 카테고리)</h4>
                    <ul>
                        <li onclick="setQuestion('퇴직연금으로 어떤 펀드에 투자하는 것이 좋을까요?')">퇴직연금으로 어떤 펀드에 투자하는 것이 좋을까요?</li>
                        <li onclick="setQuestion('안정적인 수익률을 보이는 퇴직연금 상품 추천해주세요.')">안정적인 수익률을 보이는 퇴직연금 상품 추천해주세요.</li>
                        <li onclick="setQuestion('퇴직연금 포트폴리오 구성 방법이 궁금합니다.')">퇴직연금 포트폴리오 구성 방법이 궁금합니다.</li>
                        <li onclick="setQuestion('고위험 고수익 퇴직연금 상품은 어떤 것이 있나요?')">고위험 고수익 퇴직연금 상품은 어떤 것이 있나요?</li>
                    </ul>
                </div>
            </div>
        </div>

        <script>
            // API 설정
            const API_URL = '/query/';

            document.getElementById('submit-btn').addEventListener('click', async () => {
                const question = document.getElementById('question').value.trim();
                if (!question) {
                    alert('질문을 입력해주세요.');
                    return;
                }

                // 로딩 표시
                document.getElementById('loading').style.display = 'block';
                document.getElementById('response-category').textContent = '';
                document.getElementById('response-text').textContent = '';

                try {
                    const response = await fetch(API_URL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question }),
                    });

                    if (!response.ok) {
                        throw new Error('API 호출 중 오류가 발생했습니다.');
                    }

                    const data = await response.json();

                    // 응답에서 카테고리 추출
                    let category = '';
                    let text = data.answer;

                    if (text.includes('[ASSISTANT]')) {
                        category = '일반 퇴직연금 정보';
                        text = text.replace('[ASSISTANT]', '').trim();
                    } else if (text.includes('[FUND]')) {
                        category = '퇴직연금 펀드/투자 정보';
                        text = text.replace('[FUND]', '').trim();
                    }

                    document.getElementById('response-category').textContent = category;
                    document.getElementById('response-text').textContent = text;
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById('response-text').textContent = '오류가 발생했습니다: ' + error.message;
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });

            function setQuestion(text) {
                document.getElementById('question').value = text;
                // 스크롤을 입력 섹션으로 자동 이동
                document.getElementById('question').scrollIntoView({ behavior: 'smooth' });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
