#from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

# prompt = hub.pull("rlm/rag-prompt")
# NOTE: hub is deprecated
# Error: 
# Pulling a public prompt by owner/name is disabled by default because prompts may contain untrusted serialized LangChain objects.
# > LangChain hat das Verhalten geändert, weil Prompts aus dem Hub potentiell serialisierte Objekte enthalten können.
# 
# Best practice: implement locally

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:""",
        )
    ]
)

additional_prompt_template = ChatPromptTemplate.from_template(
    template = "Take {text} and rephrase it like Yoda from StarWars."
)

# Extension of generation chain
# > get "normal" result first via the RAG-prompt
# > then transform the result using another call to the llm with the additoinal_prompt_template
generation_chain = prompt | llm | StrOutputParser() | additional_prompt_template | llm | StrOutputParser()

