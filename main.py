from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools import TavilySearchResults
from crewai.tools import tool
from crewai import Crew, Task, Agent, Process
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="groq/llama-3.3-70b-versatile", temperature=0)

file_path = "git-cheat-sheet-education.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


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
    search = TavilySearchResults(api_key=tavily_api_key, k=3)
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
    goal = 'Route user question to a vectorstore or web search',
    backstory = (
        "You are a smart routing agent. Your job is to analyze the user question"
        "and determine if the answer can be found in the provided internal documents (vectorstore),"
        "or if a web search is necessary to get the latest or broader information."
        "You are familiar with the internal documents, especially related to Git functions and commands."),
    verbose = True,
    allow_delegation = True,
    llm = llm,
    max_iterations = 1,
    memory = True
)

Vector_Search_Agent = Agent(
    role = "Vector Search Expert",
    goal = "Retrieve relevant information from the internal vectorstore to answer the question",
    backstory = (
        "You are an expert in retrieving information from the internal document collection (vectorstore)."
        "Whenever the Router Agent determines that the answer is within the internal documents, you"
        "will extract the best matching content and summarize it into a clear and concise response."),
    verbose = True,
    allow_delegation = True,
    llm = llm,
    max_iterations = 2,
    tools = [vectorstore_search_tool, answer_generator_tool],
    memory = True
)

Web_Search_Agent = Agent(
    role = "Web Researcher",
    goal = "Retrieve the most relevant and up-to-date information from the web to answer: {question}",
    backstory = (
    "You are a highly skilled web researcher, trained to rapidly explore the digital world"
    "and uncover the most relevant, accurate, and up-to-date information."
    "Your expertise lies in efficiently filtering through vast amounts of data, distinguishing"
    "credible sources from noise, and delivering valuable insights that directly contribute"
    "to answering complex questions. Your role is critical in supporting knowledge retrieval"
    "for advanced AI systems, ensuring responses are grounded in the latest available information."),
    llm = llm,
    max_iterations = 2,
    allow_delegation = True,
    verbose = True,
    tools = [web_search_tool, answer_generator_tool],
    memory = True
)

Answer_Agent = Agent(
    role = "Answer Synthesizer",
    goal = "Provide comprehensive and accurate answers to user queries",
    backstory = (
        "You are an expert in synthesizing information from various sources into clear, "
        "concise, and accurate answers. Your strength lies in understanding the nuances of "
        "user questions and crafting responses that directly address their needs. You can "
        "integrate information from both internal documents and web searches to provide "
        "the most complete and helpful answers possible."),
    llm = llm,
    max_iterations = 1,
    allow_delegation = False,
    verbose = True,
    memory = True
)

# Define Tasks - Router, Vector Search & Web Search!
router_task = Task(
    description = (
        "Analyze the user question: {question}."
        "If the question is related to 'Git functions' or content present in the provided document,"
        "then return 'vectorstore'. Otherwise, return 'websearch'."
        "Respond with only 'vectorstore' or 'websearch' — no explanations."),
    expected_output = "A single word: either 'vectorstore' or 'websearch'. No other explanation.",
    agent = Router_Agent
)

vector_retrieval_task = Task(
    description = (
        "If the router's decision was 'vectorstore', use the vectorstore_search_tool"
        "to retrieve and summarize the most relevant information to answer: {question}."),
    expected_output = "A clear and concise answer retrieved from the internal vectorstore.",
    agent = Vector_Search_Agent,
    context = [router_task]
)

web_retrieval_task = Task(
    description = (
        "If the router's decision was 'websearch', use the web_search_tool"
        "to retrieve and summarize the most relevant and up-to-date information to answer: {question}."),
    expected_output = "A clear and concise answer retrieved from web sources.",
    agent = Web_Search_Agent,
    context = [router_task]
)

final_answer_task = Task(
    description = (
        "Review the information provided by the specialist agents and create a final, comprehensive "
        "answer to '{question}'. Your task is to:\n"
        "1. Synthesize the information from either the Vector Search Expert or Web Researcher.\n"
        "2. Ensure the answer is accurate, complete, and directly addresses the user's question.\n"
        "3. Format the answer in a clear and structured way.\n"
        "4. Add any additional context or explanations that would be helpful."),
    expected_output = "The final, comprehensive answer to the user's question",
    agent = Answer_Agent,
    context = [vector_retrieval_task, web_retrieval_task]
)


rag_crew = Crew(
    agents = [Router_Agent, Vector_Search_Agent, Web_Search_Agent, Answer_Agent],
    tasks = [router_task, vector_retrieval_task, web_retrieval_task, final_answer_task],
    process = Process.sequential,
    verbose = True
)

inputs = {"question":"Who is the current president of USA?"}
result = rag_crew.kickoff(inputs=inputs)
print("✅ Final Answer:\n", result)