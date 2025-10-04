import os
from pathlib import Path
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

# --- SETUP AND CONFIGURATION ---
load_dotenv()
print("🚀 Biotessera Agent is starting up...")

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google AI API key is configured.")
except KeyError:
    print("❌ Error: Environment variable GOOGLE_API_KEY not found.")
    exit()

# --- DATA PATHS (using pathlib) ---
ROOT_DIR = Path(__file__).parent
PERSIST_DIR = ROOT_DIR / 'tesserastore_db'
print(f"🔄 Loading TesseraStore from: {PERSIST_DIR}")

# --- LOAD THE KNOWLEDGE BASE ---
try:
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embedding_function
    )
    print("✅ TesseraStore loaded successfully!")
    print(f"   Number of items in DB: {vector_db._collection.count()}")

except Exception as e:
    print(f"❌ Error loading TesseraStore: {e}")
    exit()

# --- 1. DEFINE TOOLS (our "Worker Agents") ---
def search_knowledge_base(query: str) -> str:
    """Searches the TesseraStore for relevant text chunks based on a query."""
    results = vector_db.similarity_search(query)
    return "\n---\n".join([f"Source: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}" for doc in results])

def search_osdr_database(query: str) -> str:
    """Searches the NASA Open Science Data Repository (OSDR) for raw datasets."""
    print(f"\n🔎 DataFinder searching OSDR for: '{query}'")
    
    base_url = "https://osdr.nasa.gov/bio/repo/search?q="
    search_url = base_url + urllib.parse.quote_plus(query)
    
    try:
        headers = {'User-Agent': 'NASA Space Apps Participant Bot'}
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Finding a results card
        results = soup.find_all('div', class_='search-result-row-inner')
        if not results:
            return "No datasets found in OSDR for this query."
            
        found_datasets = []
        for res in results[:3]: # Take the first 3 results
            title_tag = res.find('a')
            title = title_tag.text.strip() if title_tag else "No title found"
            link = "https://osdr.nasa.gov" + title_tag['href'] if title_tag and title_tag.has_attr('href') else "No link found"
            found_datasets.append(f"Title: {title}\nLink: {link}")
        
        return "\n---\n".join(found_datasets)
        
    except requests.exceptions.RequestException as e:
        return f"Error connecting to OSDR: {e}"
    except Exception as e:
        return f"An unexpected error occurred while searching OSDR: {e}"

tessera_miner_tool = Tool(
    name="TesseraMiner",
    func=search_knowledge_base,
    description="Use this tool to search scientific publications for information about space biology, microgravity effects, and experimental results. The query should be a detailed question in English."
)

data_finder_tool = Tool(
    name="DataFinder",
    func=search_osdr_database,
    description="Use this tool to find raw datasets in the NASA Open Science Data Repository (OSDR). The query should be a simple keyword or project name, for example 'Bion-M1' or 'osteoclast'."
)

tools = [tessera_miner_tool, data_finder_tool]
print(f"✅ Tools initialized: {[tool.name for tool in tools]}")

# --- 2. INITIALIZE THE AGENT'S BRAIN (LLM) ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
print(f"✅ Main LLM (Gemini) initialized.")

# --- 3. CREATE THE AGENT PROMPT ---
prompt_template = """
You are a specialized AI assistant for NASA called Biotessera.
Your task is to answer questions about space biology based *only* on the information provided by your tools.
When you find information, you MUST cite your sources. For TesseraMiner, cite the 'Source' title. For DataFinder, provide the 'Title' and 'Link'.

You have access to the following tools:
{tools}

To answer the user's question, you must use the following format:

Question: the user's input question
Thought: you should always think about what to do. What is the user asking for? Do I need to use one tool or multiple tools?
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer based on the observations.
Final Answer: the final, comprehensive, and well-formatted answer to the original user question, with citations.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

agent_prompt = PromptTemplate.from_template(prompt_template)

# --- 4. CREATE THE AGENT AND EXECUTOR ---
biotessera_agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(agent=biotessera_agent, tools=tools, verbose=True)
print("✅ Agent Executor created. Biotessera is ready!")

# --- 5. RUN A TEST ---
if __name__ == '__main__':
    print("\n--- Running a complex test query on the full agent ---")
    test_query = "Summarize the effects of microgravity on bone loss and check if there are any datasets related to 'Bion-M1' in OSDR."
    
    response = agent_executor.invoke({
        "input": test_query
    })

    print("\n--- Agent's Final Answer ---")
    print(response['output'])