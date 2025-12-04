import sys
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import gradio as gr
from ddgs import DDGS
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger
from pydantic import BaseModel, Field

load_dotenv(override=True)

### Loguru logger setup ###
logger.remove()

fmt = "{time} | {name} | {extra[name]} | {level} | {message}"
logger.add(
    sys.stdout,
    level="DEBUG",
    format=fmt, 
    colorize=True
)
logger.add(
    "sidekick.log",
    level="DEBUG",
    format=fmt,
    rotation="10 MB",
    retention="7 days",
    colorize=True
)

# Named loggers
log_graph = logger.bind(name="Graph")
log_worker = logger.bind(name="Worker")
log_eval = logger.bind(name="Evaluator")
log_tools = logger.bind(name="Tools")


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user, or clarification, or the assistant is stuck")
    
# Define a real-world State 
class State(TypedDict):
    messages: Annotated[list, add_messages]
    success_criteria: str
    feedback_on_work: Optional[bool]
    success_criteria_met: bool
    user_input_needed: bool
    
def ddg_search(query: str) -> List[str]:
    """Method used to query WWW with DuckDuckGo search engine"""
    results = DDGS().text(query, max_results=5)
    return [r['body'] for r in results]

# Define tools
tools = [ddg_search]
log_tools.info(f"Available tools: {tools}")  

# Initialize the LLMs
worker_llm = ChatOpenAI(model="gpt-4o-mini")
worker_llm_with_tools = worker_llm.bind_tools(tools)

evaluator_llm = ChatOpenAI(model="gpt-4o")
evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

# Worker node
def worker(state: State) -> Dict[str, Any]:
    system_message = f"""You are a helpful assistant that can use tools to complete tasks.
        You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
        This is the success criteria: {state['success_criteria']}
        You should reply either with a question for the user about his assignment or with your final response.
        If you have question for the user, you need to reply by clearly stating your question. An example might be: 
        Question: please clarify whether you want a summary or a detailed answer
        If you've finished, reply with the final answer and do not ask a question. Simply reply with the answer."""
        
    if state['feedback_on_work']:
        system_message += f"""Previously you thought you completed the assignment, but your reply was rejected because the success criteria were not met.
            Here is the feedback on why this was rejected:
            {state['feedback_on_work']}
            With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question to the user."""
            
    # Add in the system message
    found_system_message = False
    messages = state["messages"]
    for message in messages:
        if isinstance(message, SystemMessage):
            message.content = system_message
            found_system_message = True

    if not found_system_message:
        log_worker.info(f"Found system message: {found_system_message}")
        messages = [SystemMessage(content=system_message)] + messages
    
    # Invoke the LLM with tools
    response = worker_llm_with_tools.invoke(messages)
    log_worker.info(f"Worker response: {response}")
    
    # Return updated state
    return {"messages": [response]}

def worker_router(state: State) -> str:
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "evaluator"

def format_conversation(messages: List[Any]) -> str:
    conversation = "Conversation history:\n\n"
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            text = message.content or "[Tools use]"
            conversation += f"Assistant: {text}\n"
            
    return conversation

def evaluator(state: State) -> State:
    last_response = state["messages"][-1].content
    
    system_message = """
        You are an evaluator that determines if a task has been completed successfully by an Assistant.
        Assess the Assistant's last response based on the given criteria. Respond with your feedback and with a decision whether
        the success criteria are met and whether more input is needed from the user."""
        
    user_message = f"""
        You are evaluation a conversation between the User and Assistant. You decide what action to take based on the last response from
        the Assistant. The entire conversation with the Assistant, with the user's original request and all replies, is:
        {format_conversation(state["messages"])}
        
        The success criteria for this assignment is:
        {state['success_criteria']}
        
        The final response from the Assistant that you are evaluating is:
        {last_response}
        
        Respond with your feedback and decide if the success criteria are met by the response.
        Also, decide if more user input is required either because the Assistant has a question, needs clarification or seems to be stuck
        and unable to answer without help."""
        
    if state["feedback_on_work"]:
        user_message += f"""
            Note that in a prior attempt from the Assistant, you have provided this feedback: {state['feedback_on_work']}.
            If you're seeing the Assistant repeating the same mistakes, consider responding that user input is required."""
            
    evaluator_messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]
    
    eval_result = evaluator_llm_with_output.invoke(evaluator_messages)
    new_state = {
        "messages": [{"role": "assistant", "content": f"Evaluator feedback on this answer: {eval_result.feedback}"}],
        "feedback_on_work": eval_result.feedback,
        "success_criteria_met": eval_result.success_criteria_met,
        "user_input_needed": eval_result.user_input_needed
    }
    
    return new_state

def route_based_on_evaluation(state: State) -> str:
    if state['success_criteria_met'] or state['user_input_needed']:
        return "END"
    else:
        return "worker"
    
# Set up GraphBuilder with State
graph_builder = StateGraph(State)

# Add Nodes
graph_builder.add_node("worker", worker)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("evaluator", evaluator)

# Add Edges
graph_builder.add_conditional_edges("worker", worker_router, {"tools": "tools", "evaluator": "evaluator"})
graph_builder.add_edge("tools", "worker")
graph_builder.add_conditional_edges("evaluator", route_based_on_evaluation, {"END": END, "worker": "worker"})
graph_builder.add_edge(START, "worker")

# Compile and add memory
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Show the graph
display(Image(graph.get_graph().draw_mermaid_png()))

def make_thread_id() -> str:
    return str(uuid.uuid4())

async def process_message(message, success_criteria, history, thread):
    config = {"configurable": {"thread_id": thread}}
    
    state = {
        "messages": message,
        "success_criteria": success_criteria,
        "feedback_on_work": None,
        "success_criteria_met": False,
        "user_input_needed": False
    }
    
    result = await graph.ainvoke(state, config=config)
    user = {"role": "user", "content": message}
    reply = {"role": "assistant", "content": result["messages"][-2].content}
    feedback = {"role": "assistant", "content": result["messages"][-1].content}
    return history + [user, reply, feedback]

async def reset():
    return "", "", None, make_thread_id()

with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("### Sidekick Personal Coworker")
    thread = gr.State(make_thread_id())
    
    with gr.Row():
        chatbot = gr.Chatbot(label="Sidekick", height=300, type="messages")
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to sidekick")
        with gr.Row():
            success_criteria = gr.Textbox(show_label=False, placeholder="What are your success criteria?")
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")
        
    message.submit(process_message, [message, success_criteria, chatbot, thread], [chatbot])
    success_criteria.submit(process_message, [message, success_criteria, chatbot, thread], [chatbot])
    go_button.click(process_message, [message, success_criteria, chatbot, thread], [chatbot])
    reset_button.click(reset, [], [message, success_criteria, chatbot, thread])
    
demo.launch()