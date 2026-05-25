# Updated to run in conda-environment langgraph_course_studio
# > conda env create -f langgraph_course_studio.yml
#
# NOTE:
# First Step - Setup a C-RAG (corrective rag)
# > extend "normal" RAG by a reflection process
# > reflection process shall overwatch if selected chunks from the RAG-database actually are suitable
# > if reflection process (grade_document) deems chunks as unsufficient, do websearch
#
# Second Step - Self-Reflection
# > extend C-RAG by additional checks of the generated answer 
# > this shall avoid hallucinations and off-topic answers
#
# Third Step - Adaptive-RAG
# > extend Self-Reflection-C-RAG by a router node 
# > router decides whether the rage-storage shall be completely get bypassed in favor for a direct websearch

import warnings
# Added filter for the JsonPlusSerializer warning which stems from some internal langchain/langgraph library
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change*",
)

from dotenv import load_dotenv

load_dotenv()

from graph.graph import app
from graph.state import GraphState

if __name__ == "__main__":
    print("Hello Advanced RAG")

    # folgender invoke funktioniert, ist aber von der typisierung unsauber; pylance warning
    # > print(app.invoke(input={"question": "what is agent memory?"}))

    # eigener Ansatz
    question = "What is spettekaka?" #"what is agent memory?"
    # so erzeugt man TypedDict-Klassen-Objekte richtig
    inputState: GraphState = {"question":question,"generation":"","web_search":False,"documents":[]}
    # so ist es falsch
    # > inputState = GraphState("question":question,"generation":"","web_search":False,"documents":[])
    print(app.invoke(input=inputState))
    
