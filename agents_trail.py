from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
from agno.agent import Agent
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.tools.tavily import TavilyTools
from agno.models.groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

agent = Agent(model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0), markdown=True)

file_path = "git-cheat-sheet-education.pdf"
loader = PyPDFLoader(file_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
chunks = text_splitter.split_documents(data)
embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")

client = QdrantClient(path="vector_store")
collection_name = "agent-rag"

try:
    collection_info = client.get_collection(collection_name=collection_name)
except (UnexpectedResponse, ValueError):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

vector_store.add_documents(documents=chunks)
retriever = vector_store.as_retriever()

knowledge_base = LangChainKnowledgeBase(retriever=retriever)

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    knowledge=knowledge_base,
    description="Answer to the user question from the knowledge base",
    markdown=True,
    search_knowledge=True,
    tools=[TavilyTools()],
    show_tool_calls=True
)

user_query = "What is the gold rate in chennai today?"
agent.print_response(user_query, stream=True)