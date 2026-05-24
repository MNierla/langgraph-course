from typing import Any, Dict, cast
# Added Document and cast to allow casting the retrieved files to documents
from langchain_core.documents import Document

from graph.chains.retrieval_grader import retrieval_grader
# Also importe GradeDocuments just to please the pylance tool
from graph.chains.retrieval_grader import GradeDocuments
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        doc = cast(Document,d)
        score = cast(GradeDocuments,retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}