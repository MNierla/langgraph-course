from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage

# from langgraph.graph import END, MessageGraph
# On MessageGraph:
# A StateGraph where every node receives a list of messages as input and returns one or more messages as output.
# MessageGraph is a subclass of StateGraph whose entire state is a single, append-only* list of messages. 
# Each node in a MessageGraph takes a list of messages as input and returns zero or more messages as output. 
# The add_messages function is used to merge the output messages from each node into the existing list of messages in the graph's state.

# Previously we used MessageGraph instead of StateGraph
# > motivation for switching to StateGraph is the freedom to define our own node-class
# > in MessageGraph the node-class is fixed
# > in StateGraph we can define it as we want; in this case we want to use Annoted and add_messages as a reducer
from langgraph.graph import END, StateGraph
# add_messages is a reducer function
# > adds entries to messages 
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT = "reflect"
GENERATE = "generate"

# We need to have a conversation between AI and human 
# Generation node will be the AI part so we can just evoke this chain.
# Responses from generate will be AI messages (I guess).
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

# Reflection node mimics the human in this scenario
# Here we have to extract the answer and add it as a HumanMessage manually.
def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# here we have a simple predefined criterion
# later we can have a (separate) llm for dynamic decision making
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return "end"
    return "reflect"

# content in curly brackets is important to map the strings/states/whatever from should_continue to actual states!
# to make this more clear here, I've put the strings into lowercase (original is all capital)
# > without specification it can work if should_continue returns states directly (not the case in my version)
# > without specification however drawing with mermaid would fail even if should_continue returns states
builder.add_conditional_edges(GENERATE, should_continue, path_map={"end":END, "reflect":REFLECT}) 
builder.add_edge(REFLECT, GENERATE)

# compared to previous courses which used jupyter notebook, we cannot directly draw the mermaid-graph here
# we can however print it as an ascii-code and paste it into the liveviewer at:
# https://mermaid.live/
graph = builder.compile()
#print(graph.get_graph().draw_mermaid())
#graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post
                """
            )
        ]
    }
    response = graph.invoke(inputs)
    for msg in response["messages"]:
        print(f"{msg.type}: {msg.content}\n")

    #print(response)
