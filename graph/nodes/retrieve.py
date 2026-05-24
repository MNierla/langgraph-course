from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

# NOTE:
# I guess that question is passed alongside with documents to make it accessible by the follow-up parts of the graph
# In the video it is said, that returning the question is optional here. It was added for curiosity reasons.
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    # retriever.invoke does semantic search
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}