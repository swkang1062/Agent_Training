# -*- coding: utf-8 -*-
import asyncio
import os
from typing import TypedDict, Annotated, Sequence
import operator

# LangChain 관련 모듈
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 🔽 실제 사용 중인 라이브러리를 임포트합니다.
from langchain_mcp_adapters.client import MultiServerMCPClient

# This will be initialized in the main function after loading tools
model_with_tools = None

#%%
import os

GEMINI_API_KEY = "AIzaSyBkGZrOfFB4I2V0cH1XMKP5Iaax9knvhXA"

#%%


# --- LangGraph 상태 정의 ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]



#%%

def should_continue(state: AgentState) -> str:
    """
    Determines the next step for the agent.

    Checks the last message in the state:
    - If it contains tool calls, the agent should execute the tools ("continue").
    - Otherwise, the conversation can end ("end").
    - It also includes a safety stop after 10 iterations to prevent infinite loops.
    """
    last_message = state['messages'][-1]
    print("state['messages'] === ", state['messages'])

    if len(state['messages']) > 10:
        print("--- 🚦 Max iterations reached. Ending conversation. ---")
        return "end"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Correctly check for non-empty tool_calls list
        if last_message.tool_calls:
             print(f"--- 📞 Decision: Call tool(s): {[tool['name'] for tool in last_message.tool_calls]} ---")
             return "continue"

    print("--- ✅ Decision: End conversation and respond to user. ---")
    return "end"

async def call_model(state: AgentState):
    """
    Calls the LLM with the current conversation history to decide the next action
    or generate a final response.
    """
    print("--- 🧠 Calling LLM... ---")
    messages = state['messages']
    
    print("messages ============= ", messages)
    # The model (bound with tools) will either respond directly or request a tool call
    if model_with_tools:
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}
    else:
        raise ValueError("Model with tools is not initialized.")
#%%

# --- Main Asynchronous Function ---
async def main():
    """
    Sets up and runs the LangGraph agent.
    """
    # 1. Validate API Key
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")

    # 2. Initialize the Language Model
    global model_with_tools
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    print("🤖 Google Gemini client initialized.")

    # 3. Initialize the MCP Client to connect to the tool server
    # This client will manage the connection and tool discovery.
    client = MultiServerMCPClient({
        "weather_server": {
            "transport": "streamable_http",
            "url": "http://localhost:8080/mcp"
        }
    })

    # 4. Fetch tools from the remote server
    print("🔗 Connecting to server to fetch tools...")
    try:
        tools = await client.get_tools()
        if not tools:
            print("⚠️ No tools found on the server. Please ensure the server is running and provides tools.")
            return
        print(f"✅ Tools loaded successfully: {[tool.name for tool in tools]}")
    except Exception as e:
        print(f"❌ Error connecting to server or loading tools: {e}")
        print("   Please ensure the 'weather_mcp_server.py' is running on http://localhost:8080/mcp.")
        return

    # 5. Bind the fetched tools to the model
    # This allows the model to know which tools it can call.
    model_with_tools = model.bind_tools(tools)

    # 6. Define the LangGraph workflow
    workflow = StateGraph(AgentState)

    # 6a. Define nodes
    # 'agent': The node that calls the LLM.
    # 'action': The node that executes the tools.
    workflow.add_node("agent", call_model)
    workflow.add_node("action", ToolNode(tools))

    # 6b. Define edges
    # The entry point is the 'agent' node.
    workflow.set_entry_point("agent")

    # Add a conditional edge from 'agent'. Based on the `should_continue` function,
    # it will either go to 'action' (if a tool is called) or end the graph.
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # After an action is performed, the graph always goes back to the 'agent' node.
    workflow.add_edge("action", "agent")

    # 6c. Compile the graph into a runnable application
    app = workflow.compile()
    print("✅ LangGraph workflow compiled successfully.")

    # --- Interactive Chat Loop ---
    print("\n🤖 제주도 사투리 날씨 에이전트 '날씨봇'이라꽈. 무신거 물어보쿠광? (종료: 'quit')")
    

    # --- 실행 코드 ---
    async def run_agent(query: str):
        """LangGraph 에이전트를 실행하고 결과를 출력하는 함수"""
        system_message = (
        "당신은 날씨 정보를 알려주는 전문 에이전트 '날씨봇'입니다. "
        "주어진 도구를 활용하여 사용자의 질문에 대한 답을 찾아야 합니다. "
        "모든 단계를 논리적으로 수행하고, 최종 답변은 반드시 **제주도 사투리**로, "
        "정감 있고 친절하게 마무리해주세요. (예: ~했수다, ~꽈?, ~걸마씸)"
        )
        # HumanMessage만 사용하도록 수정 (시스템 메시지는 첫번째 HumanMessage에 포함)
        initial_messages = [HumanMessage(content=f"{system_message}\n\n질문: {query}")]

        final_state = await app.ainvoke({"messages": initial_messages})
        final_answer = final_state['messages'][-1]
        print("\n" + "="*50)
        print("✅ 최종 답변")
        print("="*50)
        print(final_answer.content)


    print("\n🤖 제주도 사투리 날씨 에이전트 '날씨봇'이라꽈. 무신거 물어보쿠광? (종료: 'quit')")
    while True:
        user_input = input("\n👤 질문: ")
        if user_input.lower() in ["quit", "exit", "종료"]:
            print("👋 안녕히 가십서게.")
            break
        if not user_input:
            continue
        await run_agent(user_input)

#%%


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 비동기 함수인 main()을 실행합니다.
    asyncio.run(main())

