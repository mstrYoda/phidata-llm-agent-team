from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.model.google import Gemini
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.file import FileTools
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.ollama import OllamaEmbedder
from phi.tools.googlesearch import GoogleSearch
from phi.tools.website import WebsiteTools
from phi.tools.crawl4ai_tools import Crawl4aiTools
from phi.agent import AgentKnowledge

from tools import bs4

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
    tools=[GoogleSearch(), DuckDuckGo()],
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
    tools=[Crawl4aiTools()],
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
    model=gemini_model,
    description= """
        You are a team of experts about web research and file operations.
        Your job is to work together to solve the given problem.
    """,
    role="expert team",
    instructions=["If user asks somethink, search for the most relevant information on the web.",
                  "If user asks a file operation, use the file agent and write the information you gain to file",
                  "If you face with any error return error message to the user"],
    team=[research_agent, file_agent],
    add_history_to_messages=True,
    num_history_responses=2,
    show_tool_calls=True,
    monitoring=True,
    debug_mode=True,
)

team.cli_app(markdown=True)