# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weather MCP Server using FastMCP
3개의 날씨 관련 함수를 제공하는 MCP 서버
"""

import json
import os
import sys
from datetime import datetime, timedelta
import re

import google.generativeai as genai
from tavily import TavilyClient
from fastmcp import FastMCP

### 교육용 Key
os.environ["GEMINI_API_KEY"] = "AIzaSyBkGZrOfFB4I2V0cH1XMKP5Iaax9knvhXA"

os.environ["TAVILY_API_KEY"] = "여기에  API Key"

# API 키 설정 (환경변수에서 가져오기)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
if TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
#%%

# Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)

# Tavily 클라이언트 초기화
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

#%%
# Initialize FastMCP server
mcp = FastMCP("weather_server")


#%%


@mcp.tool()

def get_specific_date(date_description: str) -> str:
    """
    자연어로 된 날짜 표현을 실제 날짜로 변환하는 함수

    Args:
        date_description: 자연어 날짜 표현 (예: "어제", "오늘", "3일 후", "내일")

    Returns:
        str: YYYY-MM-DD 형식의 날짜
    """
    print("날짜를 구하는 함수 입니다.... ================", file=sys.stderr)

    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_weekday = datetime.now().strftime("%A")

        # 기본적인 날짜 표현 처리
        date_lower = date_description.lower().strip()

        if "오늘" in date_lower or "today" in date_lower:
            return current_date
        elif "어제" in date_lower or "yesterday" in date_lower:
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "내일" in date_lower or "tomorrow" in date_lower:
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "모레" in date_lower:
            return (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        elif "글피" in date_lower:
            return (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")

        # 숫자가 포함된 상대적 날짜 처리
        # "3일 후", "5일 전" 등의 패턴
        future_match = re.search(r'(\d+)일?\s*후', date_lower)
        if future_match:
            days = int(future_match.group(1))
            return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

        past_match = re.search(r'(\d+)일?\s*전', date_lower)
        if past_match:
            days = int(past_match.group(1))
            return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # 주 단위 처리
        if "다음주" in date_lower or "next week" in date_lower:
            return (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        elif "지난주" in date_lower or "last week" in date_lower:
            return (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Gemini를 사용한 복잡한 날짜 파싱
        if GEMINI_API_KEY:
            model = genai.GenerativeModel('gemini-1.5-pro')

            prompt = f"""
            오늘은 {current_date} ({current_weekday})입니다.

            다음 날짜 표현을 YYYY-MM-DD 형식으로 변환해주세요: "{date_description}"

            정확한 날짜만 응답해주세요. 다른 설명은 필요없습니다.
            응답 형식: YYYY-MM-DD
            """

            response = model.generate_content(prompt)
            result_date = response.text.strip()

            # 날짜 형식 검증
            try:
                datetime.strptime(result_date, "%Y-%m-%d")
                return result_date
            except ValueError:
                return current_date
        else:
            return current_date

    except Exception as e:
        print(f"날짜 변환 오류: {e}", file=sys.stderr)
        return datetime.now().strftime("%Y-%m-%d")

@mcp.tool()
def verify_location(location_name: str) -> str:
    """
    지역명을 검증하고 정확한 위치 정보를 Gemini Flash 모델을 사용하여 반환하는 함수

    Args:
        location_name: 검색할 지역명 (예: "서울", "한국 수도", "파리")

    Returns:
        str: JSON 형식의 위치 정보 문자열
    """
    print(f"🛠️  verify_location (Gemini Flash) 호출: {location_name}", file=sys.stderr)
    try:
        if not GEMINI_API_KEY:
            error_info = {
                "original_name": location_name,
                "verified_name": location_name,
                "country": "Unknown",
                "found": False,
                "error": "GEMINI_API_KEY not found"
            }
            return json.dumps(error_info, ensure_ascii=False)

        # gemini-1.5-flash 모델을 사용합니다.
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        당신은 지리 정보 전문가입니다. 다음 지역명을 분석하여 표준화된 도시/지역 이름과 국가를 JSON 형식으로 반환해주세요.

        입력: "{location_name}"

        요구사항:
        1. '한국 수도', '대한민국 수도' 등은 '서울'로 표준화해주세요.
        2. 주요 광역시는 '부산', '대구' 등으로 간결하게 표현해주세요.
        3. 해외 도시의 경우, 가장 널리 알려진 이름으로 반환해주세요.
        4. 응답은 반드시 아래와 같은 JSON 형식이어야 합니다. 다른 설명은 추가하지 마세요.

        {{
          "original_name": "{location_name}",
          "verified_name": "표준화된 지역명",
          "country": "국가명 (예: 대한민국, 프랑스)"
        }}
        """

        # Gemini 모델 호출
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # 모델 응답이 마크다운(```json)을 포함할 경우 제거
        if response_text.startswith("```json"):
            response_text = response_text.strip("```json").strip()
        if response_text.endswith("```"):
            response_text = response_text.rstrip("```").strip()

        # 응답 텍스트를 JSON으로 파싱
        location_data = json.loads(response_text)
        location_data["found"] = True  # 기존 로직과의 호환성을 위해 'found' 키 추가

        return json.dumps(location_data, ensure_ascii=False)

    except Exception as e:
        # 오류 발생 시 기존과 동일한 형식의 오류 메시지 반환
        print(f"지역 검증 중 오류 발생 (Gemini): {e}", file=sys.stderr)
        error_info = {
            "original_name": location_name,
            "verified_name": location_name, # 실패 시 원래 이름 사용
            "country": "Unknown",
            "found": False,
            "error": str(e)
        }
        return json.dumps(error_info, ensure_ascii=False)

@mcp.tool()
def search_weather(location: str, date: str) -> str:
    """
    특정 지역의 특정 날짜 날씨 정보를 검색하는 함수

    Args:
        location: 지역명 (예: "서울", "부산")
        date: YYYY-MM-DD 형식의 날짜

    Returns:
        str: JSON 형식의 날씨 정보 문자열
    """
    print("날씨를 구하는 함수 입니다.... ================", file=sys.stderr)
    try:
        if not TAVILY_API_KEY:
            error_info = {
                "location": location,
                "date": date,
                "found": False,
                "summary": "Tavily API 키가 설정되지 않았습니다.",
                "error": "TAVILY_API_KEY not found"
            }
            return json.dumps(error_info, ensure_ascii=False)

        # 현재 날짜와 비교하여 검색 쿼리 조정
        current_date = datetime.now().date()
        target_date = datetime.strptime(date, "%Y-%m-%d").date()

        if target_date <= current_date:
            # 과거/현재 날씨
            search_query = f"{location} 날씨 {date} 기온 온도 weather temperature"
        else:
            # 미래 날씨 예보
            search_query = f"{location} 날씨 예보 {date} 기온 forecast weather temperature"

        print(f"🔍 검색 쿼리: {search_query}", file=sys.stderr)

        # Tavily를 사용하여 날씨 정보 검색
        search_results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )

        weather_info = {
            "location": location,
            "date": date,
            "found": False,
            "summary": "날씨 정보를 찾을 수 없습니다.",
            "details": [],
            "answer": ""
        }

        if search_results:
            weather_info["found"] = True

            # Tavily의 answer 사용 (있는 경우)
            if 'answer' in search_results and search_results['answer']:
                weather_info["answer"] = search_results['answer']
                weather_info["summary"] = search_results['answer'][:300]

            # 검색 결과 처리
            if 'results' in search_results and search_results['results']:
                weather_info["details"] = [
                    {
                        "title": result.get('title', ''),
                        "content": result.get('content', '')[:200],
                        "url": result.get('url', '')
                    }
                    for result in search_results['results'][:3]
                ]

                # 첫 번째 결과로 요약 생성 (answer가 없는 경우)
                if not weather_info["answer"]:
                    first_result = search_results['results'][0]
                    weather_info["summary"] = first_result.get('content', '')[:300]

        return json.dumps(weather_info, ensure_ascii=False, indent=2)

    except Exception as e:
        error_info = {
            "location": location,
            "date": date,
            "found": False,
            "summary": f"날씨 검색 중 오류가 발생했습니다: {str(e)}",
            "error": str(e)
        }
        return json.dumps(error_info, ensure_ascii=False)

if __name__ == "__main__":
    # FastMCP 서버 실행
    mcp.run(transport="stdio")
    
