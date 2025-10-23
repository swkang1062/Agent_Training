# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 21:35:53 2025

@author: SW Kang
"""

# langgraph_agent_mcp.py

# -*- coding: utf-8 -*-
import os
import json
import subprocess
import atexit
from getpass import getpass
from typing import Annotated, TypedDict
import operator

# LangChain 관련 라이브러리
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 1. MCP 서버 프로세스 시작 ---
print("📡 MCP 서버를 백그라운드에서 시작합니다...")
mcp_server_process = subprocess.Popen(
    ['python', 'mcp_server.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding='utf-8'
)

# 프로그램 종료 시 자식 프로세스도 함께 종료되도록 등록
@atexit.register
def cleanup():
    print("🧹 에이전트가 종료되어 MCP 서버를 정리합니다.")
    if mcp_server_process.poll() is None:
        mcp_server_process.terminate()
        mcp_server_process.wait()
    print("👋 MCP 서버가 종료되었습니다.")

# MCP 서버와 통신하는 헬퍼 함수
def call_mcp_tool(method: str, **params) -> str:
    """MCP 서버의 함수를 호출하고 결과를 반환합니다."""
    request_id = 1 # 간단한 예제이므로 ID는 고정
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": request_id,
    }
    # 요청을 JSON 문자열로 변환하여 서버의 stdin으로 전송
    mcp_server_process.stdin.write(json.dumps(request) + '\n')
    mcp_server_process.stdin.flush()

    # 서버의 stdout에서 응답을 읽음
    response_line = mcp_server_process.stdout.readline()
    if not response_line:
        return json.dumps({"error": "MCP 서버로부터 응답이 없습니다."})

    response = json.loads(response_line)
    
    # 에러가 있는지 확인
    if 'error' in response:
        return json.dumps(response['error'])

    return response.get('result', '')

# --- 2. API 키 설정 (에이전트용) ---
# 에이전트의 LLM이 사용할 API 키
    
os.environ["GEMINI_API_KEY"] = "AIzaSyBkGZrOfFB4I2V0cH1XMKP5Iaax9knvhXA"
os.environ["TAVILY_API_KEY"] = "Your Tavily Key"
    
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
print("GEMINI_API_KEY == ", GEMINI_API_KEY)

# --- 3. MCP 서버 함수를 호출하는 LangChain 도구 정의 ---
# 기존 함수들의 docstring을 그대로 사용하여 LLM이 함수의 역할을 이해하도록 합니다.
@tool
def get_specific_date(date_description: str) -> str:
    """자연어로 된 날짜 표현을 실제 날짜로 변환하는 함수"""
    return call_mcp_tool('get_specific_date', date_description=date_description)

@tool
def verify_location(location_name: str) -> str:
    """지역명을 검증하고 정확한 위치 정보를 Gemini Flash 모델을 사용하여 반환하는 함수"""
    return call_mcp_tool('verify_location', location_name=location_name)

@tool
def search_weather(location: str, date: str) -> str:
    """특정 지역의 특정 날짜 날씨 정보를 검색하는 함수"""
    return call_mcp_tool('search_weather', location=location, date=date)

# --- 4. LangGraph Agent 생성 (기존 코드 재사용) ---

# 에이전트 상태(State) 정의
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 도구 및 모델 설정
tools = [get_specific_date, verify_location, search_weather]
tool_node = ToolNode(tools)

# 사용할 모델 정의 및 도구 바인딩
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, api_key=GEMINI_API_KEY)
model = model.bind_tools(tools)

# 노드(Node) 및 엣지(Edge) 정의
def call_model(state: AgentState):
    """LLM을 호출하는 에이전트 노드"""
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Tool을 호출할지, 아니면 사용자에게 최종 답변을 할지 결정합니다."""
    # Action은 답을 얻을 때까지 최대 10번 반복
    # Human + 10*(AI+Tool)) -> 메시지 수 21
    if len(state['messages']) > 21:
        print("🚦 최대 반복 횟수(10회)에 도달하여 종료합니다.")
        return "end"
    if state['messages'][-1].tool_calls:
        return "continue"
    return "end"

# 그래프(Graph) 생성
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)
workflow.add_edge("action", "agent")
    
app = workflow.compile()

# --- 5. 실행 코드 (기존 코드 재사용) ---
def run_agent(query: str):
    """LangGraph 에이전트를 실행하고 결과를 출력하는 함수"""
    system_message = (
        "당신은 날씨 정보를 알려주는 전문 에이전트 '날씨봇'입니다. "
        "주어진 도구를 활용하여 사용자의 질문에 대한 답을 찾아야 합니다. "
        "모든 단계를 논리적으로 수행하고, 최종 답변은 반드시 **제주도 사투리**로, "
        "정감 있고 친절하게 마무리해주세요. (예: ~했수다, ~꽈?, ~걸마씸)"
    )
    initial_messages = [
        HumanMessage(content=system_message),
        HumanMessage(content=query)
    ]
    final_state = app.invoke({"messages": initial_messages})
    final_answer = final_state['messages'][-1]
    print("\n" + "="*50)
    print("✅ 최종 답변")
    print("="*50)
    print(final_answer.content)

if __name__ == "__main__":
    # MCP 서버가 부팅될 시간을 잠시 기다립니다.
    import time
    time.sleep(2)
    
    os.environ["GEMINI_API_KEY"] = "Your GEMINI API Key"
    os.environ["TAVILY_API_KEY"] = "Your Tavily API Key"
    
    
    print("\n🤖 제주도 사투리 날씨 에이전트 '날씨봇'이라꽈. 무신거 물어보쿠광? (종료: 'quit')")
    while True:
        user_input = input("\n👤 질문: ")
        if user_input.lower() in ["quit", "exit", "종료"]:
            break
        if not user_input:
            continue
        run_agent(user_input)


# 루프가 끝나면 atexit에 등록된 cleanup 함수가 자동으로 호출됩니다.
