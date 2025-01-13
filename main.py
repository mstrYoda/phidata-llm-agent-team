from phi.agent import Agent, AgentMemory
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.file import FileTools
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.ollama import OllamaEmbedder
from phi.tools.googlesearch import GoogleSearch
from phi.tools.website import WebsiteTools
from phi.tools.crawl4ai_tools import Crawl4aiTools
from phi.memory.db.sqlite import SqliteMemoryDb
from phi.memory.summarizer import MemorySummarizer
from phi.memory.classifier import MemoryClassifier
from phi.storage.agent.sqlite import SqlAgentStorage

def send_request(addr: str) -> str:
    """Sends http get request to the given address and returns the response
    
    Args:
        addr: The address to send the request to

    Returns:
        str: The response text
    """
    import requests
    response = requests.get(addr)
    return response.text

#embedder = OllamaEmbedder()
gemini_model = Gemini(id="gemini-2.0-flash-exp", api_key="")
groq_model = Groq(id="llama3-8b-8192", api_key="")

leader_memory = SqliteMemoryDb(db_file="leader_memory.db")
file_memory = SqliteMemoryDb(db_file="file_memory.db")
search_memory = SqliteMemoryDb(db_file="search_memory.db")

leader_storage = SqlAgentStorage(table_name="agent_sessions", db_file="agent_storage.db")

# db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# knowledge_base = AgentKnowledge(
#     vector_db=PgVector(table_name="trends", 
#                        db_url=db_url, 
#                        search_type=SearchType.hybrid, 
#                        embedder=embedder),
# )

# Load the knowledge base: Comment out after first run
#knowledge_base.load(recreate=True, upsert=True)

file_agent = Agent(
    model=gemini_model,
    description="""
    You are a computer file operator.
    Your job is to read and write files.
    """,
    role="file writer",
    instructions=[
        "Read and write files.", 
        "If you create file successfully, return the file name to the user.", 
        "If you face with an error, return the error to the user."],
    tools=[FileTools()],
    memory = AgentMemory(
        db=file_memory, 
        create_user_memories=True, create_session_summary=True, 
        summarizer=MemorySummarizer(model=gemini_model),
        classifier=MemoryClassifier(model=gemini_model)),
    add_history_to_messages=True,
    num_history_responses=2,
    show_tool_calls=True,
    markdown=True,
    monitoring=True,
    debug_mode=True,
)

research_agent = Agent(
    model=groq_model,
    description="""
    You are a researcher.
    Your job is to search for the most relevant information on the web and return the results.
    """,
    instructions=[
        "Search for the most relevant information on the web and return the results.",
        "If the information is not found, return that you could not found the information."],
    tools=[DuckDuckGo()],
        memory = AgentMemory(
        db=search_memory, 
        create_user_memories=True, create_session_summary=True, 
        summarizer=MemorySummarizer(model=gemini_model),
        classifier=MemoryClassifier(model=gemini_model)),
    add_history_to_messages=True,
    num_history_responses=2,
    show_tool_calls=True,
    markdown=True,
    monitoring=True,
    debug_mode=True,
)

crawler_agent = Agent(
    model=gemini_model,
    description="""
    You are a web site crawler.
    Your job is to get the content of the given website.
    """,
    instructions=[
        "Get the page content of the given web site",
        "If you can not get it, return that you could not get it."],
    #tools=[Crawl4aiTools()],
    tools=[send_request],
    add_history_to_messages=True,
    num_history_responses=2,
    show_tool_calls=True,
    markdown=True,
    monitoring=True,
    debug_mode=True,
)

#research_agent.print_response("Get me the list of good hotels in Istanbul", stream=False)
#file_agent.print_response("create me a file and put numbers from 1 to 10 in it")

team = Agent(
    model=groq_model,
    description= """
        You are a team of experts about web research, webpage crawler and file operations.
        Your job is to work together to solve the given problem.
    """,
    role="expert team",
    instructions=["If you remember the given task, return your answer from memory",
                  "If user asks somethink, search for the most relevant information on the web.",
                  "If user asks a webpage content, crawl the given page",
                  "If user asks a file operation, use the file agent and write the information you gain to file",
                  "If you face with any error return error message to the user"],
    team=[research_agent, crawler_agent, file_agent],
    add_history_to_messages=True,
    num_history_responses=2,
    session_id="f6961678-58a9-44e9-b74a-6da11add6f59",
    memory = AgentMemory(
        db=leader_memory, 
        update_user_memories_after_run=True, create_user_memories=True, create_session_summary=True, 
        summarizer=MemorySummarizer(model=gemini_model),
        classifier=MemoryClassifier(model=gemini_model)),
    storage=leader_storage,
    show_tool_calls=True,
    monitoring=True,
    debug_mode=True,
)

team.cli_app(markdown=True)