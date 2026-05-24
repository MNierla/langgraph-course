# NOTE: name of this file has to start with "test_" otherwise pytest won't find/recognize it

from dotenv import load_dotenv

load_dotenv()

from typing import cast
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
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

