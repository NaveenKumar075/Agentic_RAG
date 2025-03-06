from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from crewai_tools  import tool
from crewai import Crew
from crewai import Task
from crewai import Agent
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0)

file_path = "git-cheat-sheet-education.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_k=3)

# Actual RAG logic should come here!

web_search_tool = TavilySearchResults(k=3)
web_search_tool.run()

@tool
def router_tool(question):
  """Router Function"""
  if 'git' in question:
    return 'vectorstore'
  else:
    return 'web_search'


Router_Agent = Agent(
  role = 'Router',
  goal = 'Route user question to a vectorstore or web search',
  backstory = (
    "You are an expert at routing a user question to a vectorstore or web search."
    "Use the vectorstore for questions on concept related to Retrieval-Augmented Generation."
    "You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search."
  ),
  verbose = True,
  allow_delegation = False,
  llm = llm,
)

Retriever_Agent = Agent(
role = "Retriever",
goal = "Use the information retrieved from the vectorstore to answer the question",
backstory = (
    "You are an assistant for question-answering tasks."
    "Use the information present in the retrieved context to answer the question."
    "You have to provide a clear concise answer."
),
verbose = True,
allow_delegation = False,
llm = llm,
)

router_task = Task(
    description = ("Analyse the keywords in the question {question}"
    "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
    "Return a single word 'vectorstore' if it is eligible for vectorstore search."
    "Return a single word 'websearch' if it is eligible for web search."
    "Do not provide any other premable or explaination."
    ),
    expected_output = ("Give a binary choice 'websearch' or 'vectorstore' based on the question"
    "Do not provide any other premable or explaination."),
    agent = Router_Agent,
    tools = [router_tool],
)

retriever_task = Task(
    description = ("Based on the response from the router task extract information for the question {question} with the help of the respective tool."
    "Use the web_serach_tool to retrieve information from the web in case the router task output is 'websearch'."
    "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
),
    expected_output = ("You should analyse the output of the 'router_task'"
    "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
    "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
    "Return a claer and consise text as response."),
    agent = Retriever_Agent,
    context = [router_task],
   #tools=[retriever_tool],
)

rag_crew = Crew(
    agents = [Router_Agent, Retriever_Agent],
    tasks = [router_task, retriever_task],
    verbose = True,

)

inputs = {"question":"What are the main git functions?"}
result = rag_crew.kickoff(inputs=inputs)