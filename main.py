# First Aim - Setup a C-RAG (corrective rag)
# > extend "normal" RAG by a reflection process
# > reflection process shall overwatch if selected chunks from the RAG-database actually are suitable
# > if reflection process (grade_document) deems chunks as unsufficient, do websearch

from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    print("Hello Advanced RAG")