
#%%

# -*- coding: utf-8 -*-
import asyncio
import json
from getpass import getpass

from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack

# ⭐️ 모델 라이브러리 변경
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# fastmcp 라이브러리의 비동기 클라이언트 관련 모듈들은 그대로 사용
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

import google.generativeai as genai

#%%
import os

GEMINI_API_KEY = "AIzaSyBkGZrOfFB4I2V0cH1XMKP5Iaax9knvhXA"

#%%
class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

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
            model="gemini-2.5-pro",
            google_api_key=google_api_key
        )
        
        # Initialize session and client objects
        self.sessions: List[ClientSession] = [] # 추가
        self.exit_stack = AsyncExitStack() # 추가

        self.available_tools: List[ToolDefinition] = [] # 추가
        self.tool_to_session: Dict[str, ClientSession] = {} # 추가
        
        print("🤖 Google Gemini 클라이언트가 준비되었습니다.")
    
    
    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """특정 MCP server에 연결."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            ) # new
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) # new
            await session.initialize()
            self.sessions.append(session)
    
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
    
            for tool in tools: # 추가
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")
            
    async def connect_to_servers(self): # new
        """구성 파일 내의 모든 MCP 서버에 연결합니다."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query: str):
        """사용자 쿼리를 받아 LLM과 MCP 서버를 통해 응답을 생성합니다."""
        print("🤔 처리 중...")
        # ⭐️ LangChain 메시지 형식으로 대화를 시작합니다.
        messages = [HumanMessage(content=query)]
        
        # ⭐️ LangChain을 통한 호출
        response = self.model.invoke(
            input=messages,
            tools=self.available_tools
        )

        process_query = True
        while process_query:
            # LangChain AIMessage 응답 처리
            if response.content:
                # 텍스트 응답이 있는 경우
                print(response.content)
                if not response.tool_calls:
                    process_query = False
            
            if response.tool_calls:
                # 도구 호출이 있는 경우
                messages.append(response)  # AI 메시지 추가
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # 도구 실행
                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    
                    # 도구 실행 결과를 ToolMessage로 추가
                    tool_message = ToolMessage(
                        content=str(result.content),
                        tool_call_id=tool_call["id"]
                    )
                    messages.append(tool_message)
                
                # 다음 응답 생성
                response = self.model.invoke(
                    input=messages,
                    tools=self.available_tools
                )
                
                # 최종 응답 확인
                if response.content and not response.tool_calls:
                    print(response.content)
                    process_query = False
            else:
                process_query = False

    async def chat_loop(self):
        """대화형 채팅 루프를 실행"""
        print("\nMCP Chatbot Started!")
        print("요청 입력하거나 종료를 위해서는 'quit'을 입력하세요.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self): # new
        """AsyncExitStack을 사용하여 모든 리소스를 깔끔하게 닫습니다."""
        await self.exit_stack.aclose()

#%%

async def main():
    """챗봇의 메인 실행 함수입니다."""
    chatbot = MCP_ChatBot()
    try:
        # mcp 클라이언트와 세션은 "with"를 사용하여 초기화되지 않습니다.
        # 이전 레슨과 같이
        # 따라서 정리는 수동으로 처리해야 합니다.
        await chatbot.connect_to_servers() # new!
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup() #new!


if __name__ == "__main__":
    try:
        # 필요한 라이브러리: pip install langchain-google-genai fastmcp
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")


    
#%%