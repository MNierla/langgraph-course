# NOTE: name of this file has to start with "test_" otherwise pytest won't find/recognize it

from dotenv import load_dotenv

from pprint import pprint

load_dotenv()

from typing import cast
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    # if correct, docs should contain k document-chunks; k = 4 by default; see ingestion.py
    assert len(docs) == 4
    #print(docs)
    doc_txt = docs[1].page_content

    # Added explizit cast to satisfy pylance-type-checks
    res = cast(
    GradeDocuments,
    retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    )
    #res: GradeDocuments = retrieval_grader.invoke(
    #    {"question": question, "document": doc_txt}
    #)

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    # if correct, docs should contain k document-chunks; k = 4 by default; see ingestion.py
    assert len(docs) == 4

    # > docs[1] was selected by coincidence. could have been docs[0], docs[2] or docs[3]
    doc_txt = docs[1].page_content
    # Added explizit cast to satisfy pylance-type-checks
    # here we ask a question which we know of that it cannot be answered from the vector-database
    res = cast(
    GradeDocuments,
    retrieval_grader.invoke(
         {"question": "how to make pizaa", "document": doc_txt}
    )
    )

    #res: GradeDocuments = retrieval_grader.invoke(
    #    {"question": "how to make pizaa", "document": doc_txt}
    #)

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    
    # if correct, docs should contain k document-chunks; k = 4 by default; see ingestion.py
    assert len(docs) == 4

    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
