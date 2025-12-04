import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import gradio as gr
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from loguru import logger
import sys

load_dotenv(override=True)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def push(text: str) -> None:
    """Send a push notification to the user"""
    print(f"Message to push: {text}")
    print(f"Push successful")

tool_push = Tool(
    name="send_push_notification",
    func=push,
    description="Useful for when you want to send a push notification"
)

async_browser = create_async_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit(async_browser=async_browser)
tools = toolkit.get_tools()

all_tools = tools + [tool_push]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(all_tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

memory = MemorySaver()
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=all_tools))
graph_builder.add_conditional_edges("chatbot", tools_condition, "tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "10"}}

def chat(user_input: str, history):
    result = asyncio.run(
        graph.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]}, 
            config=config
        )
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    gr.ChatInterface(chat, type="messages").launch()