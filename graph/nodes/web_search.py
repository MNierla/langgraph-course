from typing import Any, Dict

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    #print("--- RAW TAVILY RESULT ---")
    #print(tavily_results)
    #print(type(tavily_results))
    # NOTE: Due to API changes there exists and additional layer in tavily_results now
    # Instead of a list with ["content"] we have an outer list with query, follow_up_questions, ..., and results.
    # Under results we have the desired list with ["content"]
    results = tavily_results.get("results", [])
    #print("### Tavily results ###")
    #print(results)

    # the following "tavily_results = ..." came from Revision "a450f9b"
    # > it seems like the course teacher has run into the same issues regarding the new result-structure ;)
    # tavily_results = web_search_tool.invoke({"query": question})['results']

    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in results]
    )
    web_results = Document(page_content=joined_tavily_result)
  
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    #print("### Documents ###")
    #print(documents)
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    # > hier sollte "documents": None durch "docuemnts":[] ersetzt werden (leere Liste)
    # > damit lässt sich zudem die pylance-type Warnung beheben
    # > weiterhin werden generation und web_search gesetzt
    web_search(state={"question": "agent memory","generation": "","web_search": True,"documents": []})