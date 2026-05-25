# NOTE:
# Every step of the RAG-progress can be optimized
# In this course only the retriever is optimized
# The embedding, chunking etc. is basic

from dotenv import load_dotenv
load_dotenv()

# check if USER_AGENT is set (needed for WebBaseLoader; has to be loaded before importing WebBaseLoader)
import os
#print(os.getenv("USER_AGENT"))

from pathlib import Path
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings

# we don't provide local files this time
# instead we download urls first via a WebBaseLoader
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0 # kein overlap?!
)
doc_splits = text_splitter.split_documents(docs_list)

# setup of vectorstrore only needs to be done once
# > therefore the following code has been outcommentend
# > to make it "nicer" let's check if ./.chroma exists. If not create it.
CHROMA_PATH = "./.chroma"

vectorstore = Chroma(
    collection_name="rag-chroma",
    persist_directory=CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(),
)

print("Dokumente im Vectorstore:", vectorstore._collection.count())

if vectorstore._collection.count() == 0:
    print("Befülle Vectorstore...")

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH,
    )

retriever = vectorstore.as_retriever()

# test -> add it to testcases!
#question = "agent memory"
#docs = retriever.invoke(question)
#
#print(docs)
#print(len(docs))
#
#question = "pizza hut"
#docs = retriever.invoke(question)
#
#print(docs)
#print(len(docs))
