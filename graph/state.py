from typing import List, TypedDict
from langchain_core.documents import Document
#from langchain.schema import Document

# NOTE:
# Compared to previous reflection/reflexion-agents we don't use MessageGraph-nodes for this project
# MessageGraph-nodes can just handle "messages"-list but we need more attributes
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    # documents contains documents from vector-storage and/or web_search
    # > changed from List[str] to List[Document]
    documents: List[Document]