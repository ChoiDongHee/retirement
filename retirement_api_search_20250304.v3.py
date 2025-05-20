from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import redis
import numpy as np
import logging
import os
import re
import json
import time
from datetime import datetime
from contextlib import contextmanager
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
from redis.exceptions import ConnectionError, AuthenticationError
from fastapi.responses import HTMLResponse, Response
import configparser

# FastAPI 앱 초기화
app = FastAPI(title="은퇴 펀드 검색 API", description="자연어 기반 벡터 펀드 검색 시스템")


# 영어 -> 한글 컬럼 매핑
COLUMN_MAPPING = {
    'fund_id': 'No.',
    'fund_name': '펀드명',
    'fund_type': '펀드유형',
    'base_price': '기준가(원)',
    'aum': '운용규모(단위: 억)',
    'setup_date': '설정일',
    'total_fee': '총보수',
    'return_3m': '수익률_3개월',
    'return_6m': '수익률_6개월',
    'return_1y': '수익률_1년',
    'return_3y': '수익률_3년',
    'std_dev': '표준편차',
    'beta': '베타',
    'sharpe': '샤프지수',
    'fee': '수수료',
    'front_fee': '선취수수료',
    'back_fee': '후취수수료',
    'company': '판매사',
    'risk_level': '위험등급',
    'description': '펀드 설명'
}

# 한글 -> 영어 필드 매핑 (역매핑)
REVERSE_COLUMN_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}

# 수익률 기간 키워드 매핑
RETURN_PERIOD_MAPPING = {
    "단기": "return_3m",
    "중기": "return_1y",
    "장기": "return_3y"
}

FUND_TYPE_KEYWORDS = {
    # 국내 펀드 유형
    "국내": "국내주식형",
    "국내 주식": "국내주식형",
    "국내주식": "국내주식형",
    "국내 주식형": "국내주식형",
    "한국 주식": "국내주식형",
    "국내 채권": "국내채권형",
    "국내채권": "국내채권형",
    "국내 혼합": "국내혼합형",

    # 해외 펀드 유형
    "해외": "해외주식형",
    "해외 주식": "해외주식형",
    "해외주식": "해외주식형",
    "해외 주식형": "해외주식형",
    "글로벌 주식": "해외주식형",
    "해외 채권": "해외채권형",
    "해외 혼합": "해외혼합형",

    # 일반 유형
    "주식형": ["주식형", "국내주식형", "해외주식형"],
    "채권형": ["채권형", "국내채권형", "해외채권형"],
    "혼합형": ["혼합형", "국내혼합형", "해외혼합형"]
}



# 위험도 키워드 매핑
RISK_LEVEL_KEYWORDS = {
    "저위험": ["낮음", "낮은 위험"],
    "중위험": ["중간", "중간 위험"],
    "고위험": ["높음", "높은 위험"]
}


# ===== 로깅 설정 =====

def setup_logger():
    """로깅 시스템 설정"""
    # 로그 디렉토리 생성
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 오늘 날짜를 파일명에 포함
    today = datetime.now().strftime('%Y%m%d')
    log_file = f'{log_dir}/fund_search_api_{today}.log'

    # 로거 설정
    logger = logging.getLogger("fund_search_api")
    logger.setLevel(logging.INFO)

    # 핸들러가 이미 있으면 추가하지 않음
    if not logger.handlers:
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# 로거 초기화
logger = setup_logger()


# ===== Redis 연결 관리 =====

class RedisConnection:
    """Redis 연결 관리 클래스"""
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisConnection, cls).__new__(cls)
            cls._instance._init_pool()
        return cls._instance

    def _init_pool(self):
        """Redis 연결 풀 초기화"""
        try:

            # ConfigParser 객체 생성 및 설정 파일 로드
            config = configparser.ConfigParser()
            config_path = os.path.join('./config', 'config.ini')

            if not os.path.exists(config_path):
                logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
                raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

            config.read(config_path, encoding='utf-8')
            logger.info("설정값 로딩 완료")

            if 'REDIS' not in config:
                logger.error("설정 파일에 REDIS 섹션이 없습니다.")
                raise KeyError("설정 파일에 REDIS 섹션이 없습니다.")

            # Redis 설정 섹션 가져오기
            redis_config = config['REDIS']

            logger.info("Redis 연결 풀 초기화 중...")
            self._pool = redis.ConnectionPool(
                host=redis_config['REDIS_HOST'],
                port=int(redis_config['REDIS_PORT']),
                db=int(redis_config['REDIS_DB']),
                password=redis_config.get('REDIS_PASSWORD'),  # 없으면 None 반환
                decode_responses=redis_config.get('REDIS_DECODE_RESPONSES', 'False').lower() == 'true',
                max_connections=int(redis_config.get('REDIS_MAX_CONNECTIONS', '10')),
                socket_timeout=int(redis_config.get('REDIS_SOCKET_TIMEOUT', '5')),
                socket_connect_timeout=int(redis_config.get('REDIS_SOCKET_CONNECT_TIMEOUT', '5')),
                retry_on_timeout=redis_config.get('REDIS_RETRY_ON_TIMEOUT', 'True').lower() == 'true'
            )
            logger.info("Redis 연결 풀 초기화 완료")
        except Exception as e:
            logger.error(f"Redis 연결 풀 초기화 실패: {e}", exc_info=True)
            raise

    def get_connection(self):
        """Redis 연결 반환"""
        if not self._pool:
            self._init_pool()
        return redis.Redis(connection_pool=self._pool)

    @contextmanager
    def get_connection_context(self):
        """컨텍스트 매니저로 Redis 연결 제공"""
        conn = None
        try:
            conn = self.get_connection()
            conn.ping()  # 연결 확인
            yield conn
        except (ConnectionError, AuthenticationError) as e:
            logger.error(f"Redis 연결 오류: {e}", exc_info=True)
            # 연결 풀 재초기화 후 재시도
            self._init_pool()
            try:
                conn = self.get_connection()
                yield conn
            except Exception as retry_e:
                logger.error(f"Redis 재연결 실패: {retry_e}", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"Redis 연결 오류: {e}", exc_info=True)
            raise
        finally:
            pass  # ConnectionPool 사용 중이므로 명시적 close 불필요


# ===== 임베딩 모델 관리 =====

class EmbeddingModel:
    """임베딩 모델 관리 클래스"""
    _instance = None
    _model = None
    _vector_dim = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """임베딩 모델 로드"""
        try:
            logger.info("임베딩 모델 로딩 시작...")
            model_name = "BAAI/bge-large-zh-v1.5"  # 다국어 지원 BGE 모델
            self._model = SentenceTransformer(model_name)
            self._vector_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"임베딩 모델 로딩 완료: {model_name} (차원: {self._vector_dim})")
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 실패: {e}", exc_info=True)
            raise

    def get_model(self):
        """모델 인스턴스 반환"""
        if not self._model:
            self._load_model()
        return self._model

    def encode(self, text):
        """텍스트를 벡터로 인코딩"""
        if not self._model:
            self._load_model()
        return self._model.encode(text)


# 싱글톤 인스턴스
redis_conn = RedisConnection()
embedding = EmbeddingModel()


# ===== 자연어 처리 함수 =====

def identify_return_period(query_text):
    """쿼리에서 수익률 기간 추출"""
    if re.search(r'단기|3개월|3달|최근', query_text):
        return "return_3m"
    elif re.search(r'장기|3년|삼년|오래', query_text):
        return "return_3y"
    else:
        # 기본값은 1년 수익률
        return "return_1y"


def parse_fund_types(query_text):
    """
    쿼리에서 펀드 유형 추출 (개선된 버전)
    더 정확한 자연어 이해를 위해 키워드 매칭 방식 개선
    """
    fund_types = []
    query_text = query_text.lower()  # 대소문자 구분 없이 처리

    # 키워드 우선순위 설정 (더 구체적인 것이 먼저 처리되도록)
    priority_keywords = [
        ("국내주식형", ["국내 주식", "국내주식", "국내 주식형", "국내주식형"]),
        ("해외주식형", ["해외 주식", "해외주식", "해외 주식형", "해외주식형", "글로벌 주식", "글로벌주식"]),
        ("국내채권형", ["국내 채권", "국내채권", "국내 채권형", "국내채권형"]),
        ("해외채권형", ["해외 채권", "해외채권", "해외 채권형", "해외채권형", "글로벌 채권", "글로벌채권"]),
        ("국내혼합형", ["국내 혼합", "국내혼합", "국내 혼합형", "국내혼합형"]),
        ("해외혼합형", ["해외 혼합", "해외혼합", "해외 혼합형", "해외혼합형", "글로벌 혼합", "글로벌혼합"]),
        ("주식형", ["주식형", "주식 펀드", "주식"]),
        ("채권형", ["채권형", "채권 펀드", "채권"]),
        ("혼합형", ["혼합형", "혼합 펀드", "혼합"])
    ]

    # 국내/해외 구분을 먼저 확인
    region_detected = False
    if any(kw in query_text for kw in ["국내", "한국", "한국형", "국내형"]):
        for fund_type, keywords in priority_keywords:
            if "국내" in fund_type:
                for keyword in keywords:
                    if keyword in query_text:
                        fund_types.append(fund_type)
                        region_detected = True
                        break

    if any(kw in query_text for kw in ["해외", "글로벌", "세계", "외국", "글로벌형", "해외형"]):
        for fund_type, keywords in priority_keywords:
            if "해외" in fund_type:
                for keyword in keywords:
                    if keyword in query_text:
                        fund_types.append(fund_type)
                        region_detected = True
                        break

    # 지역 구분이 없는 경우, 일반 유형만 확인
    if not region_detected:
        for fund_type, keywords in priority_keywords:
            if fund_type in ["주식형", "채권형", "혼합형"]:
                for keyword in keywords:
                    if keyword in query_text:
                        fund_types.append(fund_type)
                        break

    # 중복 제거 및 우선순위 적용
    result = []
    for ft in fund_types:
        if ft not in result:
            result.append(ft)

    return result
def parse_risk_levels(query_text):
    """
    쿼리에서 위험도 추출 (개선된 버전)
    """
    risk_levels = []
    query_text = query_text.lower()  # 대소문자 구분 없이 처리

    # 위험도 키워드 매핑 (더 정확한 매칭을 위해 확장)
    risk_keywords = {
        "저위험": ["저위험", "낮은 위험", "위험도 낮은", "안전한", "안정적인", "보수적인", "위험 낮은"],
        "중위험": ["중위험", "중간 위험", "위험도 중간", "적절한 위험", "중간", "균형", "위험 중간"],
        "고위험": ["고위험", "높은 위험", "위험도 높은", "공격적인", "위험 높은", "적극적인"]
    }

    for risk_level, keywords in risk_keywords.items():
        for keyword in keywords:
            if keyword in query_text:
                risk_levels.append(risk_level)
                break  # 같은 위험도 범주에서 하나만 추가

    return list(set(risk_levels))  # 중복 제거

def extract_numeric_condition(query_text):
    """쿼리에서 수치 조건 추출"""
    # 수치 조건 패턴: "1년 수익률 20% 이상", "수익률이 5% 이상인"
    patterns = [
        r"([\w\s]+)\s+(\d+\.?\d*)(%?)[\s]*(이상|초과|이하|미만|같은)?",
        r"([\w\s]+)이\s+(\d+\.?\d*)(%?)[\s]*(이상|초과|이하|미만|같은)?",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, query_text)
        if matches:
            for match in matches:
                field_text = match[0].strip()
                value = match[1].strip()
                condition = match[3] if len(match) > 3 else "이상"  # 기본값은 이상

                # 필드명 변환
                field_mapping = {
                    "3개월 수익률": "return_3m",
                    "6개월 수익률": "return_6m",
                    "1년 수익률": "return_1y",
                    "3년 수익률": "return_3y",
                    "총보수": "total_fee",
                    "운용규모": "aum",
                    "기준가": "base_price"
                }

                for key, value_field in field_mapping.items():
                    if key in field_text:
                        return {
                            "field": value_field,
                            "value": float(value),
                            "condition": condition
                        }

    return None


def parse_query(query_text):
    """쿼리 분석"""
    return {
        "return_period": identify_return_period(query_text),
        "fund_types": parse_fund_types(query_text),
        "risk_levels": parse_risk_levels(query_text),
        "numeric_condition": extract_numeric_condition(query_text)
    }


# ===== 데이터 추출 및 검색 함수 =====

def extract_hash_data(doc, redis_client, as_json=False):
    """
    Redis HASH 문서 데이터 추출 및 JSON 변환

    Args:
        doc: Redis 검색 결과 문서
        redis_client: Redis 클라이언트 인스턴스
        as_json: JSON 형식으로 직접 변환할지 여부 (기본값: False)

    Returns:
        dict 또는 str: 문서 데이터 딕셔너리 또는 JSON 문자열
    """
    # 기본 문서 데이터 추출
    doc_dict = {}
    for key in dir(doc):
        if not key.startswith('__') and key not in ['id', 'payload'] and not callable(getattr(doc, key)):
            doc_dict[key] = getattr(doc, key)

    # ID 추가
    doc_dict['id'] = doc.id

    # Redis에서 전체 HASH 데이터 가져오기
    try:
        hash_data = redis_client.hgetall(doc.id)

        if hash_data:
            logger.debug(f"HASH 데이터 추출 (ID: {doc.id}): {len(hash_data)}개 필드")

            # 수치형 필드 목록
            numeric_fields = [
                'return_1y', 'return_3y', 'return_3m', 'return_6m',
                'aum', 'total_fee', 'std_dev', 'beta', 'sharpe',
                'base_price', 'front_fee', 'back_fee'
            ]

            # 각 필드 처리
            for field_key, field_value in hash_data.items():
                # 바이트 키 처리
                if isinstance(field_key, bytes):
                    try:
                        key = field_key.decode('utf-8')
                    except UnicodeDecodeError:
                        # 다른 인코딩 시도
                        try:
                            key = field_key.decode('euc-kr')
                        except UnicodeDecodeError:
                            key = field_key.decode('latin-1')
                else:
                    key = field_key

                # 바이트 값 처리
                if isinstance(field_value, bytes):
                    try:
                        value = field_value.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            value = field_value.decode('euc-kr')
                        except UnicodeDecodeError:
                            value = field_value.decode('latin-1')
                else:
                    value = field_value

                # 수치 필드는 float으로 변환
                if key in numeric_fields and value is not None and value != '':
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"숫자 변환 실패 (필드: {key}, 값: {value})")
                        # 숫자 변환 실패 시 원래 값 유지

                # None 또는 빈 문자열 처리
                if value == '' or value is None:
                    value = None

                # 필드 값 저장
                doc_dict[key] = value

    except Exception as e:
        logger.error(f"HASH 데이터 추출 실패 (ID: {doc.id}): {e}")

    # JSON 직렬화
    if as_json:
        try:
            return json.dumps(doc_dict, ensure_ascii=False, indent=2)
        except TypeError as json_err:
            logger.warning(f"JSON 직렬화 실패 (ID: {doc.id}): {json_err}")

            # 안전한 직렬화를 위해 변환
            safe_dict = {}
            for k, v in doc_dict.items():
                try:
                    # 각 필드별로 직렬화 테스트
                    json.dumps({k: v})
                    print(json.dumps())
                    safe_dict[k] = v
                except TypeError:
                    # 직렬화 불가능한 값은 문자열로 변환
                    if v is not None:
                        safe_dict[k] = str(v)
                    else:
                        safe_dict[k] = None

            return json.dumps(safe_dict, ensure_ascii=False, indent=2)

    return doc_dict


def vector_search_with_hash(query_text, redis_client, k=10):
    """
    HASH 형태의 데이터에 대한 벡터 검색 수행

    Args:
        query_text: 검색 쿼리
        redis_client: Redis 클라이언트
        k: 반환할 결과 수

    Returns:
        list: 검색 결과 목록 (딕셔너리 형태)
    """
    start_time = time.time()
    logger.info(f"벡터 검색 시작: '{query_text}'")

    try:
        # 벡터 인코딩
        query_vector = embedding.encode(query_text)
        query_vector = query_vector.astype(np.float32)

        # 벡터 검색 쿼리 생성
        q = Query(f'*=>[KNN {k} @description_vector $BLOB]') \
            .dialect(2) \
            .return_fields('*', '__description_vector_score') \
            .paging(0, k) \
            .sort_by('__description_vector_score')

        # 검색 실행
        params = {
            'BLOB': query_vector.tobytes(),
            'K': k
        }

        results = redis_client.ft('idx:funds').search(q, params)
        logger.info(f"벡터 검색 결과: {len(results.docs)}개")

        # 결과 변환
        documents = []
        for doc in results.docs:
            try:
                # 코사인 유사도 (1 - 거리)
                similarity = 1 - float(doc.__description_vector_score)

                # HASH 데이터 추출
                doc_dict = extract_hash_data(doc, redis_client)
                doc_dict['vector_similarity'] = similarity

                documents.append(doc_dict)
            except Exception as doc_e:
                logger.error(f"문서 처리 중 오류 (ID: {doc.id}): {doc_e}")

        elapsed_time = time.time() - start_time
        logger.info(f"벡터 검색 완료 (소요 시간: {elapsed_time:.4f}초)")

        return documents

    except Exception as e:
        logger.error(f"벡터 검색 실패: {e}", exc_info=True)
        raise


def get_json_search_response(query_text, results, conditions=None):
    """
    검색 결과를 표준 JSON 응답 형식으로 변환

    Args:
        query_text: 검색 쿼리
        results: 검색 결과 목록
        conditions: 검색 조건 (옵션)

    Returns:
        str: JSON 형식의 응답
    """
    # 검색 조건이 없으면 기본값 설정
    if conditions is None:
        conditions = {
            "return_period": "return_1y",
            "fund_types": [],
            "risk_levels": [],
            "numeric_condition": None
        }

    # 응답 구성
    response = {
        "query": query_text,
        "results": results,
        "total_results": len(results),
        "conditions": conditions,
        "timestamp": datetime.now().isoformat()
    }

    # JSON으로 변환
    try:
        json_str = json.dumps(response, ensure_ascii=False, indent=2)
        return json_str
    except TypeError as e:
        logger.error(f"JSON 직렬화 실패: {e}")

        # 안전한 변환 시도
        safe_response = {
            "query": query_text,
            "results": [],  # 결과 제외
            "total_results": len(results),
            "conditions": conditions,
            "error": "JSON 직렬화 실패",
            "error_details": str(e),
            "timestamp": datetime.now().isoformat()
        }

        # 변환된 결과 하나씩 추가 (중요 필드만)
        for idx, item in enumerate(results):
            try:
                safe_item = {
                    "index": idx,
                    "id": item.get("id", f"item_{idx}"),
                    "fund_name": str(item.get("fund_name", "")),
                    "fund_type": str(item.get("fund_type", "")),
                    "return_1y": item.get("return_1y"),
                    "return_3y": item.get("return_3y"),
                    "aum": item.get("aum")
                }
                safe_response["results"].append(safe_item)
            except Exception:
                safe_response["results"].append({"index": idx, "error": "항목 직렬화 실패"})

        return json.dumps(safe_response, ensure_ascii=False, indent=2)


def apply_filters(results, conditions):
    """
    검색 결과에 필터 적용
    개선된 버전: 더 정확한 필터링을 위해 문자열 매칭 방식 개선
    """
    filtered_results = results.copy()

    # 펀드 유형 필터
    if 'fund_types' in conditions and conditions['fund_types']:
        fund_types = conditions['fund_types']
        new_filtered_results = []

        for doc in filtered_results:
            if 'fund_type' not in doc or doc['fund_type'] is None:
                continue

            fund_type_str = str(doc['fund_type']).strip()

            # 펀드 유형 정확히 매칭
            type_matched = False
            for ft in fund_types:
                # 1. 정확히 일치하는 경우
                if ft == fund_type_str:
                    type_matched = True
                    break

                # 2. 국내/해외 구분 정확히 적용
                if ft == "국내주식형" and ("국내" in fund_type_str and "주식" in fund_type_str) and "해외" not in fund_type_str:
                    type_matched = True
                    break

                if ft == "해외주식형" and ("해외" in fund_type_str and "주식" in fund_type_str):
                    type_matched = True
                    break

                # 3. 일반 유형 매칭 (문자열 포함)
                if ft in ["주식형", "채권형", "혼합형"] and ft in fund_type_str:
                    type_matched = True
                    break

            if type_matched:
                new_filtered_results.append(doc)

        filtered_results = new_filtered_results

    # 위험도 필터
    if 'risk_levels' in conditions and conditions['risk_levels']:
        risk_levels = conditions['risk_levels']
        new_filtered_results = []

        for doc in filtered_results:
            if 'risk_level' not in doc or doc['risk_level'] is None:
                continue

            risk_level_str = str(doc['risk_level']).strip()

            # 위험도 매칭
            risk_matched = False
            for rl in risk_levels:
                if rl in risk_level_str:
                    risk_matched = True
                    break

            if risk_matched:
                new_filtered_results.append(doc)

        filtered_results = new_filtered_results

    # 수치 필터
    if 'numeric_condition' in conditions and conditions['numeric_condition']:
        nf = conditions['numeric_condition']
        field, value, condition = nf['field'], nf['value'], nf['condition']
        new_filtered_results = []

        for doc in filtered_results:
            if field not in doc or doc[field] is None:
                continue

            try:
                field_value = float(doc[field])

                if condition in ["이상", "초과"] and field_value >= value:
                    new_filtered_results.append(doc)
                elif condition in ["이하", "미만"] and field_value <= value:
                    new_filtered_results.append(doc)
            except (ValueError, TypeError):
                # 숫자로 변환할 수 없는 경우 무시
                continue

        filtered_results = new_filtered_results

    # 필터링 결과가 없으면 원본 반환
    if not filtered_results and results:
        logger.warning("필터링 후 결과가 없어 원본 결과를 반환합니다.")
        return results

    return filtered_results




def sort_by_return(results, return_field):
    """수익률 기준 정렬"""
    try:
        # 수익률 필드가 있는 결과만 선택
        valid_results = [doc for doc in results if return_field in doc and doc[return_field] is not None]

        # 내림차순 정렬 (높은 수익률 순)
        sorted_results = sorted(valid_results, key=lambda x: float(x[return_field]), reverse=True)

        # 정렬된 결과가 없으면 원본 반환
        if not sorted_results:
            logger.warning(f"유효한 {return_field} 값이 없어 정렬을 건너뜁니다.")
            return results

        return sorted_results
    except Exception as e:
        logger.error(f"정렬 중 오류: {e}", exc_info=True)
        return results


# ===== API 모델 =====

class SearchQuery(BaseModel):
    """검색 요청 모델"""
    query: str
    count: Optional[int] = 5
    format: Optional[str] = "json"  # 'json' 또는 'dict'


# ===== API 엔드포인트 =====

@app.post("/api/search")
async def api_search(query: SearchQuery):
    """
    자연어 검색 API 엔드포인트

    - **query**: 자연어 검색어 (예: "높은 수익률의 저위험 펀드")
    - **count**: 반환할 결과 수 (기본값: 5)

    **특징:**
    - 펀드 유형, 위험도, 수익률 등 다양한 조건 자동 추출
    - 벡터 검색을 통한 의미 기반 검색
    - 수익률 기준 정렬 (높은 순)
    """
    try:
        # 검색 조건 분석
        conditions = parse_query(query.query)
        logger.info(f"분석된 검색 조건: {conditions}")

        # Redis 연결
        with redis_conn.get_connection_context() as r:
            # 벡터 검색 실행
            vector_results = vector_search_with_hash(query.query, r, k=query.count * 2)
            print("================")
            print(vector_results)
            print("================")
            # 필터 적용
            filtered_results = apply_filters(vector_results, conditions)
            print("================")
            print(vector_results)
            print("================")
            # 수익률 기준 정렬
            return_field = conditions['return_period']
            sorted_results = sort_by_return(filtered_results, return_field)

            # 상위 k개 결과 선택
            final_results = sorted_results[:query.count]

            # 결과가 없는 경우 대체 검색
            if not final_results:
                logger.warning(f"결과가 없어 기본 수익률 기준 검색을 실행합니다")
                try:
                    default_query = Query("*").sort_by(return_field, True).limit(0, query.count)
                    fallback_results = r.ft("idx:funds").search(default_query)

                    fallback_docs = []
                    for doc in fallback_results.docs:
                        doc_dict = extract_hash_data(doc, r)
                        fallback_docs.append(doc_dict)

                    final_results = fallback_docs
                except Exception as fallback_e:
                    logger.error(f"대체 검색 실패: {fallback_e}", exc_info=True)

            # 응답 구성
            response = {
                "query": query.query,
                "results": final_results,
                "total_results": len(final_results),
                "conditions": conditions,
                "return_period": conditions['return_period'],
                "timestamp": datetime.now().isoformat()
            }

            return response

    except Exception as e:
        logger.error(f"API 검색 오류: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}"
        )


@app.post("/api/search/json")
async def api_search_json(query: SearchQuery):
    """
    자연어 검색 API 엔드포인트 (JSON 응답)

    - **query**: 자연어 검색어 (예: "높은 수익률의 해외 주식형 펀드")
    - **count**: 반환할 결과 수 (기본값: 5)
    - **format**: 응답 형식 ('json' 또는 'dict', 기본값: 'json')

    **특징:**
    - Redis HASH 데이터 추출 및 안전한 변환
    - JSON 형식으로 응답
    """
    try:
        # 검색 조건 분석
        conditions = parse_query(query.query)
        logger.info(f"분석된 검색 조건: {conditions}")

        # Redis 연결
        with redis_conn.get_connection_context() as r:
            # 벡터 검색 실행
            vector_results = vector_search_with_hash(query.query, r, k=query.count * 2)

            # 필터 적용
            filtered_results = apply_filters(vector_results, conditions)

            # 수익률 기준 정렬
            return_field = conditions['return_period']
            sorted_results = sort_by_return(filtered_results, return_field)

            # 상위 k개 결과 선택
            final_results = sorted_results[:query.count]

            if query.format.lower() == 'json':
                # JSON 문자열 응답
                json_response = get_json_search_response(query.query, final_results, conditions)
                return Response(
                    content=json_response,
                    media_type="application/json"
                )
            else:
                # 딕셔너리 응답
                dict_response = {
                    "query": query.query,
                    "results": final_results,
                    "total_results": len(final_results),
                    "conditions": conditions,
                    "return_period": conditions['return_period']
                }
                return dict_response

    except Exception as e:

        logger.error(f"API 검색 오류: {e}", exc_info=True)
        # 오류 응답
        error_response = {
            "error": str(e),
            "query": query.query,
            "results": [],
            "total_results": 0
        }

        if query.format.lower() == 'json':
            error_json = json.dumps(error_response, ensure_ascii=False)
            return Response(
                content=error_json,
                status_code=500,
                media_type="application/json"
            )
        else:
            return HTTPException(
                status_code=500,
                detail=error_response
            )

    # 디버깅용 단일 펀드 조회 API
@app.get("/api/fund/{fund_id}")
async def get_fund(fund_id: str):
        """
        단일 펀드 정보 조회 API

        - **fund_id**: 펀드 ID (예: "fund:123")

        Redis에서 해당 ID의 펀드 정보를 HASH 형태로 조회하여 반환합니다.
        """
        try:
            with redis_conn.get_connection_context() as r:
                # 펀드 ID에 "fund:" 접두어가 없으면 추가
                if not fund_id.startswith("fund:"):
                    fund_id = f"fund:{fund_id}"

                # Redis에서 HASH 데이터 조회
                hash_data = r.hgetall(fund_id)

                if not hash_data:
                    return HTTPException(
                        status_code=404,
                        detail=f"Fund not found: {fund_id}"
                    )

                # 임시 문서 객체 생성 (ID 속성 포함)
                class TempDoc:
                    def __init__(self, id_value):
                        self.id = id_value

                temp_doc = TempDoc(fund_id)

                # 데이터 추출 및 변환
                fund_data = extract_hash_data(temp_doc, r)

                return fund_data

        except Exception as e:
            logger.error(f"펀드 조회 오류 (ID: {fund_id}): {e}", exc_info=True)
            return HTTPException(
                status_code=500,
                detail=f"펀드 조회 중 오류 발생: {str(e)}"
            )

@app.get("/api/debug/redis")
async def debug_redis():
        """
        Redis 연결 및 인덱스 정보 디버깅 API

        Redis 서버 연결 상태와 인덱스 정보를 확인합니다.
        """
        try:
            with redis_conn.get_connection_context() as r:
                # Redis 정보 조회
                info = r.info()

                # 인덱스 정보 조회
                try:
                    index_info = r.ft('idx:funds').info()
                    index_stats = {
                        "num_docs": index_info.get("num_docs", 0),
                        "fields": index_info.get("fields", []),
                        "index_name": index_info.get("index_name", ""),
                        "index_options": index_info.get("index_options", []),
                        "index_definition": index_info.get("index_definition", {})
                    }
                except Exception as idx_e:
                    index_stats = {"error": str(idx_e)}

                # 펀드 ID 목록 조회 (최대 10개)
                try:
                    keys = r.keys("fund:*")[:10]
                    fund_ids = [k.decode('utf-8') if isinstance(k, bytes) else k for k in keys]
                except Exception as keys_e:
                    fund_ids = {"error": str(keys_e)}

                response = {
                    "redis_version": info.get("redis_version", ""),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", ""),
                    "index_stats": index_stats,
                    "sample_fund_ids": fund_ids,
                    "timestamp": datetime.now().isoformat()
                }

                return response

        except Exception as e:
            logger.error(f"Redis 디버깅 오류: {e}", exc_info=True)
            return HTTPException(
                status_code=500,
                detail=f"Redis 연결 확인 중 오류 발생: {str(e)}"
            )

@app.get("/html", response_class=HTMLResponse)
async def get_simple_search_html():
        """간단한 검색 페이지 HTML 반환"""
        html_content = """
 <!DOCTYPE html>
<html>
<head>
    <title>펀드 검색</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #1a73e8;
        }
        h1 {
            margin-bottom: 20px;
        }
        h2 {
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.5rem;
        }
        .search-container {
            display: flex;
            margin-bottom: 20px;
        }
        #searchInput {
            flex: 1;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px 0 0 4px;
        }
        #searchButton {
            padding: 10px 20px;
            background-color: #1a73e8;
            color: white;
            border: none;
            border-radius: 0 4px 4px 0;
            cursor: pointer;
            font-size: 16px;
        }
        #searchButton:hover {
            background-color: #1557b0;
        }
        #searchInfo {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .fund-card {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .fund-card h3 {
            margin-top: 0;
            color: #1a73e8;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .fund-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 10px;
        }
        .stat-group {
            flex: 1;
            min-width: 200px;
        }
        .fund-description {
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }
        .return-value {
            font-weight: bold;
        }
        .return-positive {
            color: #0b8043;
        }
        .return-negative {
            color: #d50000;
        }
        .loading {
            text-align: center;
            padding: 20px;
            font-style: italic;
            color: #666;
        }
        .error {
            color: #d50000;
            font-weight: bold;
        }
        .samples-container {
            margin-top: 30px;
            margin-bottom: 40px;
        }
        .samples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }
        .sample-category {
            margin-bottom: 25px;
        }
        .sample-item {
            background-color: #f5f8ff;
            border: 1px solid #c7d8ff;
            border-radius: 4px;
            padding: 10px 15px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .sample-item:hover {
            background-color: #e8f0ff;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .sample-title {
            font-weight: bold;
            color: #1a73e8;
            margin-bottom: 5px;
        }
        .sample-description {
            color: #666;
            font-size: 0.9em;
        }
        .section-divider {
            border-top: 1px solid #eee;
            margin: 40px 0;
        }
        .filter-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }
        .filter-badge {
            background-color: #e8f0ff;
            color: #1a73e8;
            border: 1px solid #c7d8ff;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .filter-badge:hover {
            background-color: #d0e2ff;
        }
    </style>
</head>
<body>
    <h1>은퇴 펀드 검색</h1>

    <div class="search-container">
        <input type="text" id="searchInput" placeholder="예: 수익률이 높은 해외 주식형 펀드, 안정적인 채권형 펀드...">
        <button id="searchButton">검색</button>
    </div>

    <div class="filter-badges">
        <span class="filter-badge" onclick="applyFilter('유형')">펀드 유형</span>
        <span class="filter-badge" onclick="applyFilter('수익률')">수익률</span>
        <span class="filter-badge" onclick="applyFilter('기간')">투자 기간</span>
        <span class="filter-badge" onclick="applyFilter('위험')">위험도</span>
        <span class="filter-badge" onclick="applyFilter('규모')">운용 규모</span>
    </div>

    <div id="searchInfo" style="display: none;"></div>

    <div id="results"></div>

    <div class="section-divider"></div>

    <div class="samples-container">
        <h2>검색 샘플</h2>
        <p>다음 예시를 클릭하여 바로 검색해보세요:</p>

        <div class="sample-category">
            <h3>펀드 유형별 검색</h3>
            <div class="samples-grid">
                <div class="sample-item" onclick="useSearchSample('해외 주식형 펀드 추천')">
                    <div class="sample-title">해외 주식형 펀드 추천</div>
                    <div class="sample-description">해외 시장에 투자하는 주식형 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('국내 주식형 펀드 중 수익률이 좋은 펀드')">
                    <div class="sample-title">국내 주식형 고수익 펀드</div>
                    <div class="sample-description">국내 시장에 투자하며 수익률이 높은 주식형 펀드를 찾습니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('혼합형 펀드 중 안정적인 것')">
                    <div class="sample-title">안정적인 혼합형 펀드</div>
                    <div class="sample-description">주식과 채권에 함께 투자하는 안정적인 혼합형 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('채권형 펀드 저위험')">
                    <div class="sample-title">저위험 채권형 펀드</div>
                    <div class="sample-description">위험도가 낮은 채권형 펀드를 검색합니다.</div>
                </div>
            </div>
        </div>

        <div class="sample-category">
            <h3>수익률 기준 검색</h3>
            <div class="samples-grid">
                <div class="sample-item" onclick="useSearchSample('1년 수익률 15% 이상인 펀드')">
                    <div class="sample-title">높은 1년 수익률</div>
                    <div class="sample-description">1년 수익률이 15% 이상인 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('3년 수익률이 가장 높은 펀드')">
                    <div class="sample-title">최고 3년 수익률</div>
                    <div class="sample-description">장기적으로 높은 성과를 보인 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('최근 3개월 수익률이 좋은 해외 펀드')">
                    <div class="sample-title">단기 성과 해외 펀드</div>
                    <div class="sample-description">최근 3개월 동안 높은 성과를 보인 해외 펀드를 찾습니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('꾸준한 수익률의 안정적인 펀드')">
                    <div class="sample-title">꾸준한 수익률 펀드</div>
                    <div class="sample-description">변동성이 적고 안정적인 수익을 제공하는 펀드를 검색합니다.</div>
                </div>
            </div>
        </div>

        <div class="sample-category">
            <h3>투자 성향별 검색</h3>
            <div class="samples-grid">
                <div class="sample-item" onclick="useSearchSample('저위험 안정적인 은퇴 펀드')">
                    <div class="sample-title">보수적 투자자용</div>
                    <div class="sample-description">위험을 최소화하고 안정적인 수익을 원하는 투자자를 위한 펀드입니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('중위험 중수익 균형 투자 펀드')">
                    <div class="sample-title">균형 투자자용</div>
                    <div class="sample-description">적절한 위험과 수익의 균형을 맞춘 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('고위험 고수익 적극적 투자 펀드')">
                    <div class="sample-title">공격적 투자자용</div>
                    <div class="sample-description">높은 수익을 위해 위험을 감수할 수 있는 투자자를 위한 펀드입니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('장기 투자 은퇴 설계 펀드')">
                    <div class="sample-title">장기 은퇴 계획용</div>
                    <div class="sample-description">은퇴를 위한 장기 자산 구축에 적합한 펀드를 검색합니다.</div>
                </div>
            </div>
        </div>

        <div class="sample-category">
            <h3>특정 조건 검색</h3>
            <div class="samples-grid">
                <div class="sample-item" onclick="useSearchSample('운용규모 1000억 이상 대형 펀드')">
                    <div class="sample-title">대형 펀드</div>
                    <div class="sample-description">운용규모가 큰 안정적인 대형 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('총보수 0.5% 이하 저비용 펀드')">
                    <div class="sample-title">저비용 펀드</div>
                    <div class="sample-description">수수료가 낮아 실질 수익률에 유리한 펀드를 찾습니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('설정일 5년 이상 된 검증된 펀드')">
                    <div class="sample-title">장기 운용 펀드</div>
                    <div class="sample-description">오랜 기간 운용되어 검증된 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('샤프지수가 높은 효율적인 펀드')">
                    <div class="sample-title">효율적 투자 펀드</div>
                    <div class="sample-description">위험 대비 수익률이 효율적인 펀드를 검색합니다.</div>
                </div>
            </div>
        </div>

        <div class="sample-category">
            <h3>복합 조건 검색</h3>
            <div class="samples-grid">
                <div class="sample-item" onclick="useSearchSample('해외주식형 중 1년 수익률 20% 이상이고 위험등급이 중간인 펀드')">
                    <div class="sample-title">고수익 중위험 해외주식</div>
                    <div class="sample-description">높은 수익률과 적절한 위험의 해외 주식 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('국내 채권형 중 총보수 0.6% 이하이고 3년 수익률이 안정적인 펀드')">
                    <div class="sample-title">저비용 안정 채권형</div>
                    <div class="sample-description">비용이 적고 안정적 수익을 내는 채권형 펀드를 찾습니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('혼합형 펀드 중 운용규모 500억 이상이고 장기 투자에 적합한 펀드')">
                    <div class="sample-title">대형 장기 혼합형</div>
                    <div class="sample-description">규모가 크고 장기 투자에 적합한 혼합형 펀드를 검색합니다.</div>
                </div>
                <div class="sample-item" onclick="useSearchSample('베타가 낮고 샤프지수가 높은 효율적인 투자 펀드')">
                    <div class="sample-title">시장 비연동 효율 펀드</div>
                    <div class="sample-description">시장 변동성에 덜 영향받고 효율적인 수익을 내는 펀드를 검색합니다.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 검색 버튼 클릭 시 이벤트
        document.getElementById('searchButton').addEventListener('click', function() {
            performSearch();
        });

        // 엔터 키 입력 시 검색 실행
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch();
            }
        });

        // 검색 샘플 사용 함수
        function useSearchSample(sampleQuery) {
            document.getElementById('searchInput').value = sampleQuery;
            performSearch();
            // 검색 결과로 스크롤
            document.getElementById('searchInfo').scrollIntoView({ behavior: 'smooth' });
        }

        // 필터 적용 함수
        function applyFilter(filterType) {
            let sampleQuery = '';
            
            switch(filterType) {
                case '유형':
                    const types = ['해외 주식형', '국내 주식형', '혼합형', '채권형'];
                    sampleQuery = types[Math.floor(Math.random() * types.length)] + ' 펀드';
                    break;
                case '수익률':
                    const returns = ['수익률이 높은', '수익률 상위', '1년 수익률 좋은', '3년 수익률 우수한'];
                    sampleQuery = returns[Math.floor(Math.random() * returns.length)] + ' 펀드';
                    break;
                case '기간':
                    const periods = ['단기', '중기', '장기', '3개월', '1년', '3년'];
                    sampleQuery = periods[Math.floor(Math.random() * periods.length)] + ' 투자 펀드';
                    break;
                case '위험':
                    const risks = ['저위험', '중위험', '고위험', '안정적인', '공격적인'];
                    sampleQuery = risks[Math.floor(Math.random() * risks.length)] + ' 펀드';
                    break;
                case '규모':
                    const sizes = ['소형', '중형', '대형', '운용규모 큰', '운용규모 작은'];
                    sampleQuery = sizes[Math.floor(Math.random() * sizes.length)] + ' 펀드';
                    break;
            }
            
            document.getElementById('searchInput').value = sampleQuery;
            performSearch();
        }

        // 검색 실행 함수
        function performSearch() {
            // 검색어 가져오기
            const query = document.getElementById('searchInput').value.trim();

            if (!query) {
                alert('검색어를 입력해주세요.');
                return;
            }

            // 검색 정보와 결과 영역 초기화
            document.getElementById('searchInfo').style.display = 'none';
            document.getElementById('results').innerHTML = '<div class="loading">검색 중...</div>';

            // API 호출
            fetch('/api/search/json', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    count: 10,
                    format: 'json'
                }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('서버 응답 오류');
                }
                return response.json();
            })
            .then(data => {
                // 검색 정보 표시
                const infoElement = document.getElementById('searchInfo');

                // 검색 조건 정보
                let conditionsText = '';
                if (data.conditions) {
                    const returnPeriod = data.conditions.return_period || 'return_1y';
                    const periodMap = {
                        'return_3m': '3개월',
                        'return_6m': '6개월',
                        'return_1y': '1년',
                        'return_3y': '3년'
                    };

                    conditionsText = `<strong>수익률 기간:</strong> ${periodMap[returnPeriod] || '1년'}<br>`;

                    if (data.conditions.fund_types && data.conditions.fund_types.length > 0) {
                        conditionsText += `<strong>펀드 유형:</strong> ${data.conditions.fund_types.join(', ')}<br>`;
                    }

                    if (data.conditions.risk_levels && data.conditions.risk_levels.length > 0) {
                        conditionsText += `<strong>위험도:</strong> ${data.conditions.risk_levels.join(', ')}<br>`;
                    }

                    if (data.conditions.numeric_condition) {
                        const nc = data.conditions.numeric_condition;
                        const fieldMap = {
                            'return_3m': '3개월 수익률',
                            'return_6m': '6개월 수익률',
                            'return_1y': '1년 수익률',
                            'return_3y': '3년 수익률',
                            'total_fee': '총보수',
                            'aum': '운용규모',
                            'base_price': '기준가'
                        };
                        conditionsText += `<strong>수치 조건:</strong> ${fieldMap[nc.field] || nc.field} ${nc.value}% ${nc.condition}<br>`;
                    }
                }

                infoElement.innerHTML = `
                    <p><strong>검색어:</strong> ${data.query}</p>
                    <p><strong>검색 결과:</strong> ${data.total_results}개</p>
                    ${conditionsText}
                    <p><strong>검색 시간:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
                `;
                infoElement.style.display = 'block';

                // 결과 영역
                const resultsContainer = document.getElementById('results');

                // 결과가 없는 경우
                if (!data.results || data.results.length === 0) {
                    resultsContainer.innerHTML = '<p>검색 결과가 없습니다.</p>';
                    return;
                }

                // 결과 표시
                let resultsHTML = '';

                data.results.forEach((fund, index) => {
                    // 수익률 클래스 결정 (양수: 녹색, 음수: 빨간색)
                    const getReturnClass = (value) => {
                        if (!value && value !== 0) return '';
                        return parseFloat(value) >= 0 ? 'return-positive' : 'return-negative';
                    };

                    // 수익률 포맷팅
                    const formatReturn = (value) => {
                        if (!value && value !== 0) return '-';
                        return `<span class="${getReturnClass(value)}">${parseFloat(value).toFixed(2)}%</span>`;
                    };

                    // 운용규모 포맷팅
                    const formatAum = (value) => {
                        if (!value && value !== 0) return '-';
                        return `${parseFloat(value).toLocaleString()}억원`;
                    };

                    resultsHTML += `
                        <div class="fund-card">
                            <h3>${index + 1}. ${fund.fund_name || '이름 없음'}</h3>

                            <div class="fund-stats">
                                <div class="stat-group">
                                    <p><strong>펀드 유형:</strong> ${fund.fund_type || '-'}</p>
                                    <p><strong>위험 등급:</strong> ${fund.risk_level || '-'}</p>
                                    <p><strong>판매사:</strong> ${fund.company || '-'}</p>
                                </div>

                                <div class="stat-group">
                                    <p><strong>3개월 수익률:</strong> ${formatReturn(fund.return_3m)}</p>
                                    <p><strong>1년 수익률:</strong> ${formatReturn(fund.return_1y)}</p>
                                    <p><strong>3년 수익률:</strong> ${formatReturn(fund.return_3y)}</p>
                                </div>

                                <div class="stat-group">
                                    <p><strong>운용규모:</strong> ${formatAum(fund.aum)}</p>
                                    <p><strong>설정일:</strong> ${fund.setup_date || '-'}</p>
                                    <p><strong>총보수:</strong> ${fund.total_fee ? fund.total_fee + '%' : '-'}</p>
                                </div>
                            </div>

                            ${fund.description ? 
                                `<div class="fund-description"><strong>설명:</strong> ${fund.fund_summary}</div>` : ''}

                            <div style="text-align: right; margin-top: 10px; font-size: 0.9em;">
                                ${fund.vector_similarity ? 
                                    `<span>유사도: ${(fund.vector_similarity * 100).toFixed(1)}%</span>` : ''}
                            </div>
                        </div>
                    `;
                });
                resultsContainer.innerHTML = resultsHTML;
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('results').innerHTML = '<p class="error">검색 중 오류가 발생했습니다.</p>';
            });
        }
    </script>
</body>
</html>
            """
        return HTMLResponse(content=html_content)



    # 서버 실행 코드
if __name__ == "__main__":
    import uvicorn


    logger.info("은퇴 펀드 검색 서버 시작")
    try:
        # Redis 연결 및 모델 로딩 확인
        with redis_conn.get_connection_context() as r:
            logger.info("Redis 연결 확인 성공")

        model = embedding.get_model()
        logger.info("임베딩 모델 로딩 확인 성공")

        try:
            index_info = r.ft('idx:funds').info()
            print("인덱스 정보:", index_info)
            # 벡터 필드 찾기
            vector_fields = [k for k in index_info if isinstance(k, str) and 'vector' in k.lower()]
            print("발견된 벡터 필드:", vector_fields)
        except Exception as e:
            print(f"인덱스 정보 확인 실패: {e}")
            vector_field = "fund_summary_vector"  # 기본값 사용
        # 서버 실행
        uvicorn.run(app, host="0.0.0.0", port=8003)
    except Exception as e:
        logger.error(f"서버 시작 실패: {e}", exc_info=True)

