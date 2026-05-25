from dotenv import load_dotenv
from typing import cast
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    # note that the llm here is imported from react where tools got bound to it
    # cast added as well as llm only returns BaseMessages which are too generic for AnyMessage which are required by MessagesState
    response = cast(AIMessage,llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]]))
    return {"messages": [response]}

# Remember ToolNode
# > it checks whether the last message of the dialogue is an AI-message
# > if it is an AI-message and it contains a tool-call, then it will be executed by the ToolNode
tool_node = ToolNode(tools)