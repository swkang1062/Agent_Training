
#%%

# -*- coding: utf-8 -*-
import asyncio
import json
from getpass import getpass
from typing import List

# ⭐️ 모델 라이브러리 변경
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# fastmcp 라이브러리의 비동기 클라이언트 관련 모듈들은 그대로 사용
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


#%%
import os

### 교육용 Key
GEMINI_API_KEY = "AIzaSyBkGZrOfFB4I2V0cH1XMKP5Iaax9knvhXA"
#%%

class MCP_ChatBot:
    """
    MCP 서버와 연동하여 Google Gemini 모델을 통해 동작하는 비동기 챗봇 클라이언트입니다.
    """

    def __init__(self):
        # ⭐️ Google API 키를 안전하게 입력받아 클라이언트를 초기화합니다.
        google_api_key = GEMINI_API_KEY
        if not google_api_key:
            raise ValueError("API 키가 입력되지 않았습니다.")
        
        # ⭐️ Gemini 모델을 설정합니다. (향후 gemini-2.5-flash로 변경 가능)
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key
        )
        self.model_with_tools = None # 도구가 바인딩된 모델
        
        # MCP 서버 세션 및 사용 가능한 도구를 초기화합니다.
        self.session: ClientSession = None
        self.available_tools: List[dict] = []
        print("🤖 Google Gemini 클라이언트가 준비되었습니다.")

    async def process_query(self, query: str):
        """사용자 쿼리를 받아 LLM과 MCP 서버를 통해 응답을 생성합니다."""
        print("🤔 처리 중...")
        # ⭐️ LangChain 메시지 형식으로 대화를 시작합니다.
        messages = [HumanMessage(content=query)]

        # 최대 5번의 도구 호출을 허용합니다.
        for _ in range(5):
            if not self.model_with_tools:
                raise RuntimeError("모델에 도구가 설정되지 않았습니다. 서버 연결을 확인하세요.")

            # ⭐️ 모델을 비동기적으로 호출합니다.
            ai_response: AIMessage = await self.model_with_tools.ainvoke(messages)
            messages.append(ai_response)

            # ⭐️ 도구 호출(tool_calls)이 없으면, 응답을 출력하고 루프를 종료합니다.
            
            print("ai_response == ", ai_response)
            
            if not ai_response.tool_calls:
                print(f"\n💬 챗봇:\n{ai_response.content}")
                break
            
            # ⭐️ 도구 호출이 있으면 MCP 서버를 통해 실행합니다.
            tool_results = []
            for tool_call in ai_response.tool_calls:
                
                                 
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"🛠️  도구 호출: {tool_name} (인수: {tool_args})")
                

                result = await self.session.call_tool(tool_name, arguments=tool_args)

                print("result == ", result)
                # ⭐️ 결과를 LangChain의 ToolMessage 형식으로 추가합니다.
                tool_results.append(
                    ToolMessage(content=result.content, tool_call_id=tool_id)
                )
            
            messages.extend(tool_results)

    async def chat_loop(self):
        """사용자와 상호작용하는 대화 루프를 실행합니다."""
        print("\n✅ MCP 챗봇이 시작되었습니다!")
        print("질문을 입력하시거나 'quit'를 입력하여 종료하세요.")

        while True:
            try:
                query = await asyncio.to_thread(input, "\n🧑‍💻 당신의 질문: ")
                if query.strip().lower() == 'quit':
                    print("👋 챗봇을 종료합니다.")
                    break
                if not query.strip():
                    continue
                await self.process_query(query)
            except (KeyboardInterrupt, EOFError):
                print("\n👋 챗봇을 종료합니다.")
                break
            except Exception as e:
                print(f"\n🚨 오류 발생: {e}")

    async def connect_to_server_and_run(self):
        """MCP 서버에 연결하고, 도구를 설정한 후, 챗봇 루프를 시작합니다."""
        
        ############# stdio ################
        server_params = StdioServerParameters(
            command="python", # 'uv run' 대신 'python'을 사용하도록 변경
            args=["mcp_research_server.py"], # 서버 파일 이름 변경
        )
        print(f"🔌 서버 실행 시도: {' '.join([server_params.command] + server_params.args)}")
        print("server_params == ", server_params)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
        
                response = await session.list_tools()
                tools = response.tools
        
                print("\n🔗 서버 연결 성공! 사용 가능한 도구:", [tool.name for tool in tools])
        
                # ⭐️ LangChain의 .bind_tools()가 이해하는 형식으로 도구 정보를 저장합니다.
                # 'input_schema'를 'parameters'로 키 이름을 변경합니다.
                self.available_tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in tools]
        
                print("self.available_tools == ", self.available_tools)
                # ⭐️ Gemini 모델에 도구를 바인딩합니다.
                self.model_with_tools = self.model.bind_tools(self.available_tools)
                
                print("tools are bounded ... ")
        
                await self.chat_loop()
        ################################################        
        
        
        await self.chat_loop()
#%%

async def main():
    """챗봇의 메인 실행 함수입니다."""
    try:
        chatbot = MCP_ChatBot()
        await chatbot.connect_to_server_and_run()
    except ValueError as e:
        print(f"🚨 초기화 오류: {e}")
    except Exception as e:
        print(f"🚨 예기치 않은 오류로 프로그램을 종료합니다: {e}")

if __name__ == "__main__":
    try:
        # 필요한 라이브러리: pip install langchain-google-genai fastmcp
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")


    
#%%