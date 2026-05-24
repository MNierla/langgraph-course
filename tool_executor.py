from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
# Notes on ToolNode
# (class) ToolNode
# A node that runs the tools called in the last AIMessage.
# It can be used either in StateGraph with a "messages" state key (or a custom key passed via ToolNode's 'messages_key'). If multiple tool calls are requested, they will be run in parallel. The output will be a list of ToolMessages, one for each tool call.
# Tool calls can also be passed directly as a list of ToolCall dicts.
# > basically toolnode looks into the latest ai-message in messages for tool calls
# > if a tool call exists, it will be executed
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

# tavily_tool is already a useable tool which we could bind to an llm, but we do it differently here
# > we use tavily_tool in two StructuredTools below to distinguish between AnswerQuestion and ReviseAnswer
# > what is this good for? It gives clearer output in the trace!
tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
