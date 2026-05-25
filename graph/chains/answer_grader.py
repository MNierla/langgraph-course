from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
#from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(temperature=0)
# method="function_calling" added to disable the Warning:
# UserWarning: Cannot use method='json_schema' with model gpt-3.5-turbo since it doesn't support OpenAI's Structured Output API. You can see supported models here: https://platform.openai.com/docs/guides/structured-outputs#supported-models. 
# To fix this warning, set `method='function_calling'. Overriding to method='function_calling'.
structured_llm_grader = llm.with_structured_output(GradeAnswer,method="function_calling")

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# Why is it defined as RunnableSequence?
# > Compare retrieval_grader: retrieval_grader = grade_prompt | structured_llm_grader
# Using RunnableSequence here gives typing warnings from pylance
# > answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
answer_grader = answer_prompt | structured_llm_grader