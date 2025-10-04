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

from tools import get_tools

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

# --- 1. INITIALIZE TOOLS ---
tools = get_tools(vector_db)
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
    response = agent_executor.invoke({"input": test_query})
    print("\n--- Agent's Final Answer ---")
    print(response['output'])