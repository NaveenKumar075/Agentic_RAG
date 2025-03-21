from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from crewai.tools import tool
from crewai import Crew, Task, Agent
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="groq/llama-3.3-70b-versatile", temperature=0)

file_path = "git-cheat-sheet-education.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})


# Actual RAG logic should come here!
@tool("vectorstore_search_tool")
def vectorstore_search_tool(query: str) -> str:
    """
    Search the internal document vectorstore for relevant content matching the query.
    Returns a string containing the concatenated content of the top matching documents.
    """
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])

@tool("web_search_tool")
def web_search_tool(query: str) -> str:
    """
    Perform a web search using Tavily to retrieve up-to-date information matching the query.
    Returns a string containing summarized search results.
    """
    search = TavilySearchResults(api_key=tavily_api_key, k=1)
    return search.run(query)

@tool("answer_generator_tool")
def answer_generator_tool(question: str, retrieved_content: str) -> str:
    """
    Generate a comprehensive answer based on the retrieved content.
    Returns a well-formatted, concise answer to the user's question.
    """
    prompt = f"""
    Based on the following retrieved content, generate a comprehensive answer to the question.
    
    Question: {question}
    
    Retrieved Content:
    {retrieved_content}
    
    Answer:
    """
    return llm.invoke(prompt).content

# Define Agents - Router, Vector Search & Web Search!
Router_Agent = Agent(
    role = 'Router',
    goal = 'Route user question to a vectorstore or websearch.',
    backstory = (
        "You analyze the user question and determine if the answer is within internal documents (vectorstore) "
        "or requires a web search. If the question involves Git functions, you choose 'vectorstore', "
        "otherwise, you choose 'websearch'."),
    verbose = True,
    allow_delegation = True,
    llm = llm,
    max_iterations = 1,
    memory = False
)

Vector_Search_Agent = Agent(
    role = "Vector Search Expert",
    goal = "Retrieve relevant information from the internal vectorstore to answer the question.",
    backstory = ("You specialize in retrieving information from the vectorstore when the router agent selects 'vectorstore'."),
    verbose = True,
    allow_delegation = False,
    llm = llm,
    max_iterations = 1,
    tools = [vectorstore_search_tool, answer_generator_tool],
    memory = False
)

Web_Search_Agent = Agent(
    role = "Web Researcher",
    goal = "Retrieve the most relevant and up-to-date information from the web.",
    backstory = ("You specialize in searching the web when the router agent selects 'websearch'."),
    llm = llm,
    verbose = True,
    allow_delegation = False,
    max_iterations = 1,
    tools = [web_search_tool, answer_generator_tool],
    memory = False
)

Answer_Agent = Agent(
    role = "Answer Synthesizer",
    goal = "Provide comprehensive and accurate answers to user queries.",
    backstory = ("You synthesize information from different sources to generate a structured and well-formatted response."),
    llm = llm,
    verbose = True,
    allow_delegation = False,
    max_iterations = 1,
    memory = True
)

# Define Tasks - Router, Vector Search & Web Search!
router_task = Task(
    description = (
        "Analyze the user question: {question}.\n"
        "If the question is related to 'Git functions' or content present in the provided document, "
        "then return 'vectorstore'. Otherwise, return 'websearch'.\n"
        "Respond with only 'vectorstore' or 'websearch' — no explanations."),
    expected_output = "A single word: either 'vectorstore' or 'websearch'. No other explanation.",
    agent = Router_Agent
)

vector_retrieval_task = Task(
    description = ("Use vectorstore_search_tool to retrieve and summarize the most relevant information to answer: {question}."),
    expected_output = "A clear and concise answer retrieved from the internal vectorstore.",
    agent = Vector_Search_Agent,
    context = [router_task]
)

web_retrieval_task = Task(
    description = ("Use web_search_tool to retrieve and summarize the most relevant and up-to-date information to answer: {question}."),
    expected_output = "A clear and concise answer retrieved from web sources.",
    agent = Web_Search_Agent,
    context = [router_task]
)

final_answer_task = Task(
    description = (
        "Synthesize the most relevant information from the selected retrieval task (either vectorstore or websearch) "
        "to answer '{question}'. Format the response clearly and ensure it fully addresses the user's query."),
    expected_output = "A well-structured and complete answer to the user's question.",
    agent = Answer_Agent,
    context = [vector_retrieval_task, web_retrieval_task]
)

# Crews Assembling!
rag_crew = Crew(
    agents = [Router_Agent, Vector_Search_Agent, Web_Search_Agent, Answer_Agent],
    tasks = [router_task, vector_retrieval_task, web_retrieval_task, final_answer_task],
    # process = Process.sequential,
    verbose = True
)

inputs = {"question":"Who is the current president of USA?"}
result = rag_crew.kickoff(inputs=inputs)
print("✅ Final Answer:\n", result)