
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

        self.exit_stack = AsyncExitStack() # 추가

        # Tools list required for Anthropic API
        self.available_tools = []
        # Prompts list for quick display
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}
# 추가
        
        print("🤖 Google Gemini 클라이언트가 준비되었습니다.")
    
    
    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """특정 MCP server에 연결."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            # print("read == ", read)
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            print("Starting Initializing..... ")
            try:
                # List available tools
                # print(".... 1")
                response = await session.list_tools()
                # print("response list == ", response)
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                # print("..... 2")
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # print("..... 3")
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session

            except Exception as e:
                print(f"Error {e}")

        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")
            
            
            
    async def connect_to_servers(self): # new
        """구성 파일 내의 모든 MCP 서버에 연결합니다."""
        try:
            with open("server_config_3.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                print("server_name == ", server_name)
                print("server_config == ", server_config)
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query: str):
        """사용자 쿼리를 받아 LLM과 MCP 서버를 통해 응답을 생성합니다."""
        print("🤔 처리 중...")
        # ⭐️ LangChain 메시지 형식으로 대화를 시작합니다.
        messages = [HumanMessage(content=query)]

        print("messages == ", messages)
        
        while True:
            # ⭐️ LangChain을 통한 호출
            response = self.model.invoke(
                input=messages,
                tools=self.available_tools
            )

            has_tool_use = False

            # 텍스트 응답 처리
            if response.content:
                print(response.content)

            # 도구 호출 처리
            if response.tool_calls:
                has_tool_use = True
                # AI 응답을 메시지에 추가
                messages.append(response)
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]
                    
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # Get session and call tool
                    session = self.sessions.get(tool_name)
                    if not session:
                        print(f"Tool '{tool_name}' not found.")
                        break

                    # 도구 실행
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    
                    # 도구 실행 결과를 ToolMessage로 추가
                    tool_message = ToolMessage(
                        content=str(result.content),
                        tool_call_id=tool_call_id
                    )
                    messages.append(tool_message)

            # 도구 사용이 없으면 루프 종료
            if not has_tool_use:
                break
            
            
    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return

        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")

            
    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)

        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break

        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return

        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")
            
    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return

        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content

                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item)
                                  for item in prompt_content)

                print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            print(f"Error: {e}")
            
    async def chat_loop(self):
        print("\nMCP 챗봇이 시작되었습니다!")
        print("질문을 입력하거나 'quit'을 눌러 종료하세요.")
        print("사용 가능한 주제를 보려면 @folders를 사용하세요.")
        print("해당 주제의 논문을 검색하려면 @<topic>를 사용하세요.")
        print("사용 가능한 프롬프트를 나열하려면 /prompts를 사용하세요.")
        print("프롬프트를 실행하려면 /prompt <name> <arg1=value1>을 사용하세요.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue

                if query.lower() == 'quit':
                    break

                # Check for @resource syntax first
                if query.startswith('@'):
                    # Remove @ sign
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue

                # Check for /command syntax
                if query.startswith('/'):
                    parts = query.split()
                    command = parts[0].lower()

                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue

                        prompt_name = parts[1]
                        args = {}

                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value

                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue

                await self.process_query(query)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
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
        await chatbot.cleanup() 


if __name__ == "__main__":
    try:
        # 필요한 라이브러리: pip install langchain-google-genai fastmcp
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")


    
#%%