import redis
import json
from sentence_transformers import SentenceTransformer
import pandas as pd

# Redis 연결 설정
r = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    password='donghee',
    decode_responses=True
)

# BGE 모델 로드 (펀드 상세 정보를 벡터화)
model_name = "BAAI/bge-large-zh-v1.5"
model = SentenceTransformer(model_name)




# 기존 엑셀 파일 불러오기f
# file_path = "./fund_data_updated.xlsx"
# df = pd.read_excel(file_path)

# # UTF-8 인코딩을 적용하여 CSV로 저장 후 다시 엑셀로 변환
# csv_temp_path = "fund_data_updated_utf8.csv"
# excel_utf8_path = "fund_data_updated_utf8.xlsx"
#
# # CSV로 저장 (UTF-8 인코딩)
# df.to_csv(csv_temp_path, index=False, encoding="utf-8-sig")
#
# # CSV 파일을 다시 UTF-8 인코딩으로 엑셀로 저장
# df_utf8 = pd.read_csv(csv_temp_path, encoding="utf-8-sig")
# df_utf8.to_excel(excel_utf8_path, index=False)
#
# print(f"✅ UTF-8 인코딩된 엑셀 파일이 저장되었습니다: {excel_utf8_path}")



# 엑셀 파일 불러오기
file_path = "./fund_data_updated_utf8.xlsx"
df = pd.read_excel(file_path)

# 데이터 Redis에 삽입
for index, row in df.iterrows():
    fund_id = str(row["No."])  # 고유한 펀드 ID (Redis 키 값)

    # 펀드 상세정보를 벡터로 변환
    fund_summary_vector = model.encode(row["펀드상세정보"]).tolist()

    # 데이터 구조
    fund_data = {
        "fund_code": row["협회펀드코드"],
        "company": row["운용사명"],
        "homepage": row["홈페이지URL"],
        "fund_name": row["한글펀드명"],
        "fund_type": row["펀드 유형"],
        "base_price": row["기준가"],
        "aum": row["설정액"],
        "setup_date": row["설정일(텍스트)"],
        "total_fee": row["총보수"],
        "sale_fee": row["판매보수"],
        "extra_fee": row["기타 보수"],
        "return_3m": row["수익률_3개월"],
        "return_6m": row["수익률_6개월"],
        "return_1y": row["수익률_1년"],
        "return_3y": row["수익률_3년"],
        "std_dev": row["표준편차"],
        "beta": row["베타"],
        "sharpe": row["샤프지수"],
        "return_rank": row["수익률 순위"],
        "sharpe_rank": row["샤프지수 순위"],
        "risk_level": row["위험등급"],
        "seller": row["판매사"],
        "description": row["펀드 설명"],
        "keywords": row["키워드 추출(펀드설명)"],
        "fund_summary": row["펀드상세정보"],  # 일반 필드로 저장
        "fund_summary_vector": json.dumps(fund_summary_vector)  # 벡터로 저장
    }

    # Redis에 데이터 저장 (해시맵 형식)
    r.hset(f"fund:{fund_id}", mapping=fund_data)

print("✅ 모든 펀드 데이터가 Redis에 성공적으로 삽입되었습니다!")
