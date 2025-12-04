from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
import gradio as gr
import uuid
import asyncio
from dotenv import load_dotenv

import logging
from datetime import datetime

from ddgs import DDGS

# ---- Configure logging ----
logging.basicConfig(
    level=logging.DEBUG,    # change to INFO to reduce noise
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),             # console output
        logging.FileHandler("sidekick.log")  # log file
    ]
)

log_graph = logging.getLogger("Graph")
log_worker = logging.getLogger("Worker")
log_eval = logging.getLogger("Evaluator")
log_tools = logging.getLogger("Tools")
log_ui = logging.getLogger("UI")

load_dotenv(override=True)


# Fix uvloop conflict
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# -----------------------
# Evaluator model schema
# -----------------------
class EvaluatorOutput(BaseModel):
    feedback: str
    success_criteria_met: bool
    user_input_needed: bool

# -----------------------
# State for LangGraph
# -----------------------
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]   # list of LangChain messages
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool

# -----------------------
# Tools
# -----------------------
# async_browser = create_async_playwright_browser(headless=True)
# await async_browser.start()
# toolkit = PlayWrightBrowserToolkit(async_browser=async_browser)
# tools = toolkit.get_tools()

def ddg_search(query: str) -> List[str]:
    """Method used to query WWW with DuckDuckGo search engine

    Args:
        query (str): Query to be searched via DuckDuckGo

    Returns:
        List[str]: List of found results
    """
    results = DDGS().text(query, max_results=5)
    return [r['body'] for r in results]

tools = [ddg_search] 

# -----------------------
# LLMs
# -----------------------
worker_llm = ChatOpenAI(model="gpt-4o-mini")
worker_llm_with_tools = worker_llm.bind_tools(tools)
evaluator_llm = ChatOpenAI(model="gpt-4o")
evaluator_llm_struct = evaluator_llm.with_structured_output(EvaluatorOutput)

# -----------------------
# Worker node
# -----------------------
def worker(state: State) -> Dict[str, Any]:
    log_worker.debug(f"Worker node START. Feedback={state.get('feedback_on_work')}")
    log_worker.debug(f"Incoming messages: {state['messages']}")

    system_message = SystemMessage(
        content=(
            f"You are a helpful assistant that uses tools.\n"
            f"Success criteria: {state['success_criteria']}\n"
        )
    )

    if state.get("feedback_on_work"):
        system_message.content += f"\nPrevious feedback: {state['feedback_on_work']}\n"

    messages = [system_message] + state["messages"]

    log_worker.debug(f"Final message stack sent to LLM: {messages}")

    response = worker_llm_with_tools.invoke(messages)

    log_worker.debug(f"Worker LLM output: {response}")

    return {"messages": [response]}


# Route to evaluator or tool node
def worker_router(state: State) -> str:
    last = state["messages"][-1]
    log_graph.debug(f"Routing decision based on last message: {last}")

    if hasattr(last, "tool_calls") and last.tool_calls:
        log_graph.debug("→ Routing to TOOLS node")
        return "tools"

    log_graph.debug("→ Routing to EVALUATOR node")
    return "evaluator"


# Format convo for evaluator
def format_conversation(messages: List[Any]) -> str:
    out = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            out += f"User: {m.content}\n"
        elif isinstance(m, AIMessage):
            out += f"Assistant: {m.content}\n"
    return out

# -----------------------
# Evaluator node
# -----------------------
def evaluator(state: State) -> State:
    log_eval.debug("Evaluator node START")
    last = state["messages"][-1]
    log_eval.debug(f"Last assistant response: {last.content}")

    conv = format_conversation(state["messages"])
    log_eval.debug(f"Full conversation for evaluator:\n{conv}")

    system_msg = SystemMessage(content="Evaluate if criteria met.")
    user_msg = HumanMessage(
        content=(
            f"Conversation:\n{conv}\n\n"
            f"Criteria:\n{state['success_criteria']}\n\n"
            f"Last response:\n{last.content}\n"
        )
    )

    result = evaluator_llm_struct.invoke([system_msg, user_msg])

    log_eval.debug(f"Evaluator output: {result}")

    feedback_msg = AIMessage(content=result.feedback)

    return {
        "messages": [feedback_msg],
        "feedback_on_work": result.feedback,
        "success_criteria_met": result.success_criteria_met,
        "user_input_needed": result.user_input_needed,
    }
    
def route_based_on_evaluation(state: State) -> str:
    if state['success_criteria_met'] or state['user_input_needed']:
        return "END"
    else:
        return "worker"


# -----------------------
# Build graph
# -----------------------
graph_builder = StateGraph(State)

graph_builder.add_node("worker", worker)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("evaluator", evaluator)

graph_builder.add_conditional_edges("worker", worker_router, {"tools": "tools", "evaluator": "evaluator"})
graph_builder.add_edge("tools", "worker")
graph_builder.add_conditional_edges("evaluator", route_based_on_evaluation, {"END": END, "worker": "worker"})

graph_builder.add_edge(START, "worker")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# -----------------------
# Gradio Integration
# -----------------------
def make_thread_id():
    return str(uuid.uuid4())

import time

async def process_message(message, criteria, history, thread_id):
    log_ui.info(f"[thread={thread_id}] User message received: {message}")
    log_ui.info(f"[thread={thread_id}] Criteria: {criteria}")

    config = {"configurable": {"thread_id": thread_id}}

    state = {
        "messages": [HumanMessage(content=message)],
        "success_criteria": criteria,
        "feedback_on_work": None,
        "success_criteria_met": False,
        "user_input_needed": False
    }

    log_graph.debug(f"[thread={thread_id}] Starting ainvoke with state: {state}")

    start = time.time()
    result = await graph.ainvoke(state, config=config)
    duration = time.time() - start

    log_graph.info(
        f"[thread={thread_id}] Graph execution finished in {duration:.2f}s"
    )
    log_graph.debug(f"[thread={thread_id}] Graph final state: {result}")

    msgs = result["messages"]

    reply = msgs[-2].content if len(msgs) > 1 else "(no reply)"
    feedback = msgs[-1].content

    log_ui.debug(f"[thread={thread_id}] Reply: {reply}")
    log_ui.debug(f"[thread={thread_id}] Feedback: {feedback}")

    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
        {"role": "assistant", "content": feedback},
    ]


async def reset():
    new_id = make_thread_id()
    log_ui.info(f"UI reset. New thread_id={new_id}")
    return "", "", [], new_id

with gr.Blocks() as demo:
    thread = gr.State(make_thread_id())
    chatbot = gr.Chatbot(type="messages")
    message = gr.Textbox(placeholder="Your request")
    criteria = gr.Textbox(placeholder="Success criteria")

    btn_go = gr.Button("Go")
    btn_reset = gr.Button("Reset", variant="stop")

    btn_go.click(process_message, [message, criteria, chatbot, thread], chatbot)
    btn_reset.click(reset, outputs=[message, criteria, chatbot, thread])

demo.launch()
