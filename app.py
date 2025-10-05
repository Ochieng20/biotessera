import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import urllib.parse
import requests
from bs4 import BeautifulSoup
import zipfile
import requests

# --- LangChain Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Biotessera", page_icon="🧩")
st.title("🧩 Biotessera: Space Biology AI")
st.caption("An AI assistant for researching NASA's space biology publications")

# --- 2. SETUP AND CONFIGURATION ---
load_dotenv()
print("🚀 Biotessera Agent is starting up...")

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google AI API key is configured.")
except KeyError:
    print("❌ Error: Environment variable GOOGLE_API_KEY not found.")
    exit()

# --- 3. DATA PATHS (using pathlib) ---
ROOT_DIR = Path(__file__).parent
PERSIST_DIR = ROOT_DIR / 'tesserastore_db'
print(f"🔄 Loading TesseraStore from: {PERSIST_DIR}")
ZIP_PATH = ROOT_DIR / 'tesserastore_db.zip'

GOOGLE_DRIVE_FILE_ID = "1qyy8frw4AM7lGjtsLIkMnhNrqLSuzxUg"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&id=" + id
    session = requests.Session()
    response = session.get(URL, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

if not PERSIST_DIR.exists():
    with st.spinner(f"One-time setup: Downloading knowledge base..."):
        download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, ZIP_PATH)
    
    with st.spinner("Unpacking knowledge base..."):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(ROOT_DIR)
        os.remove(ZIP_PATH)

    st.success("Knowledge base is ready! The app will now load.")
    st.rerun()

# --- 4. LOAD THE KNOWLEDGE BASE ---
@st.cache_resource(show_spinner="Loading knowledge base (TesseraStore)...")
def load_vector_db():
    """Loads the Chroma vector database from the persistent directory."""
    persist_dir = str(Path(__file__).parent / 'tesserastore_db')
    if not os.path.exists(persist_dir):
        return None, f"⚠️ Database directory not found: `{persist_dir}`. Ensure it is in the same folder as app.py"
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
        msg = f"✅ Knowledge base loaded. Records: {vectordb._collection.count()}"
        return vectordb, msg
    except Exception as e:
        return None, f"❌ Error loading TesseraStore: {e}"

vector_db, load_message = load_vector_db()
st.info(load_message)
if vector_db is None:
    st.warning("TesseraMiner will be disabled.")

# --- 5. DEFINE TOOLS ---
def search_knowledge_base(query: str) -> str:
    """Searches the TesseraStore for relevant and diverse text chunks using MMR."""
    if vector_db is None:
        return "TesseraMiner is disabled: Vector DB not available."
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 20}
    )
    results = retriever.invoke(query)
    if not results:
        return "No information found in TesseraStore for this query."
    return "\n---\n".join([f"Source: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}" for doc in results])

def search_osdr_database(query: str) -> str:
    """Searches the NASA Open Science Data Repository (OSDR) for raw datasets."""
    base_url = "https://osdr.nasa.gov/bio/repo/search?q="
    search_url = base_url + urllib.parse.quote_plus(query)
    try:
        headers = {'User-Agent': 'NASA Space Apps Participant Bot'}
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='search-result-row-inner')
        if not results:
            return "No datasets found in OSDR for this query."
        found_datasets = []
        for res in results[:3]:
            title_tag = res.find('a')
            if title_tag:
                title = title_tag.text.strip()
                link = "https://osdr.nasa.gov" + (title_tag['href'] if title_tag.has_attr('href') else '')
                found_datasets.append(f"Title: {title}\nLink: {link}")
        return "\n---\n".join(found_datasets) if found_datasets else "No datasets found in OSDR for this query."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to OSDR: {e}"
    except Exception as e:
        return f"An unexpected error occurred while searching OSDR: {e}"

tessera_miner_tool = Tool(
    name="TesseraMiner",
    func=search_knowledge_base,
    description="Use this tool to search for information about space biology, microgravity effects, and related scientific research. The query should be a detailed question in English."
)

# --- 6. INITIALIZE THE AGENT'S BRAIN (LLM) ---
@st.cache_resource(show_spinner="Building AI agent...")
def build_agent_executor():
    """Builds the LangChain agent and executor."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1)

    tools = []
    if vector_db is not None:
        tools.append(Tool(
            name="TesseraMiner",
            func=search_knowledge_base,
            description="Use this tool to search scientific publications for information about space biology, microgravity effects, and experimental results. The query should be a detailed question in English."
        ))
    tools.append(Tool(
        name="DataFinder",
        func=search_osdr_database,
        description="Use this tool to find raw datasets in the NASA Open Science Data Repository (OSDR). The query should be a simple keyword or project name, for example 'Bion-M1' or 'proteomics'."
    ))
    print(f"✅ Tools initialized: {[tool.name for tool in tools]}")

    prompt_template = """
    You are a specialized AI assistant for NASA called Biotessera.
    Your task is to answer questions about space biology based *only* on the information provided by your tools.
    Think step-by-step. If the user asks for both research and datasets, use the appropriate tool for each part of the query.
    ***VERY IMPORTANT RULE***: If you find yourself using the exact same tool with the exact same input twice in a row, you are stuck in a loop. You MUST either try a different tool or significantly change your Action Input. Do not repeat the same action again.
    **Crucially, you MUST cite your sources. For TesseraMiner, cite the 'Source' title. For DataFinder, provide the 'Title' and 'Link'.**

    You have access to the following tools:
    {tools}

    To answer the user's question, you must use the following format:

    Question: the user's input question
    Thought: you should always think about what to do.
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

    # --- 7. CREATE THE AGENT AND EXECUTOR ---
    agent = create_react_agent(llm, tools, agent_prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=8,
        early_stopping_method="force",
        handle_parsing_errors=True,
        return_intermediate_steps=True # Key for showing agent's thoughts
    )
    return executor, [tool.name for tool in tools]

agent_executor, available_tools = build_agent_executor()
st.sidebar.title("Agent Status")
st.sidebar.success(f"Agent is ready with tools: {', '.join(available_tools)}")

# --- 8. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask your question about space biology..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Biotessera is analyzing the request... 🧠"):
            try:
                response = agent_executor.invoke({"input": prompt})
                response_text = response.get("output", "An error occurred, no answer found.")
                intermediate_steps = response.get("intermediate_steps", [])
            except Exception as e:
                response_text = f"❌ Agent error: {e}"
                intermediate_steps = []

        st.markdown(response_text)

        if intermediate_steps:
            with st.expander("🧭 See agent's thought process"):
                for i, (action, observation) in enumerate(intermediate_steps, 1):
                    st.markdown(f"**Step {i}: Using `{action.tool}`**")
                    st.code(action.tool_input, language="text")
                    st.text("Observation:")
                    st.code(observation, language='text')
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response_text})