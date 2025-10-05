import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import urllib.parse
import requests
from bs4 import BeautifulSoup

# --- LangChain Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

# --- SETUP ---
load_dotenv()
st.set_page_config(page_title="Biotessera (Test)", page_icon="🧩")
st.title("🧩 Biotessera (Minimal Test)")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ Error: Environment variable GOOGLE_API_KEY not found.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to configure Google Generative AI: {e}")
    st.stop()

ROOT_DIR = Path(__file__).parent
PERSIST_DIR = ROOT_DIR / "tesserastore_db"
st.caption(f"🔄 TesseraStore path: `{PERSIST_DIR}`")

# --- CACHED RESOURCES ---
@st.cache_resource(show_spinner=False)
def load_vector_db(persist_dir: Path):
    if not persist_dir.exists():
        return None, "⚠️ Vector database folder not found, TesseraMiner will be disabled."
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedding_function
        )
        ok = f"✅ TesseraStore loaded. Items: {vectordb._collection.count()}"
        return vectordb, ok
    except Exception as e:
        return None, f"❌ TesseraStore loading error: {e}"

vector_db, load_msg = load_vector_db(PERSIST_DIR)
st.write(load_msg)

# --- TOOLS ---
def search_knowledge_base(query: str) -> str:
    if vector_db is None:
        return "TesseraMiner disabled: vector DB not available."
    results = vector_db.similarity_search(query)
    return "\n---\n".join(
        [f"Source: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}" for doc in results]
    )

def search_osdr_database(query: str) -> str:
    print(f"\n🔎 DataFinder searching OSDR for: '{query}'")
    base_url = "https://osdr.nasa.gov/bio/repo/search?q="
    search_url = base_url + urllib.parse.quote_plus(query)
    try:
        headers = {"User-Agent": "NASA Space Apps Participant Bot"}
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("div", class_="search-result-row-inner")
        if not results:
            return "No datasets found in OSDR for this query."
        found = []
        for res in results[:3]:
            title_tag = res.find("a")
            title = title_tag.text.strip() if title_tag else "No title found"
            link = "https://osdr.nasa.gov" + title_tag["href"] if title_tag and title_tag.has_attr("href") else "No link found"
            found.append(f"Title: {title}\nLink: {link}")
        return "\n---\n".join(found)
    except requests.exceptions.RequestException as e:
        return f"Error connecting to OSDR: {e}"
    except Exception as e:
        return f"An unexpected error occurred while searching OSDR: {e}"

MAX_TOOL_CALLS = 3
tool_calls = {"TesseraMiner": 0, "DataFinder": 0}

def tool_guard(name, fn):
    def wrapped(q: str):
        tool_calls[name] += 1
        if tool_calls[name] > MAX_TOOL_CALLS:
            return f"[{name}] Call limit reached. Please synthesize final answer."
        return fn(q)
    return wrapped

tools = []
if vector_db is not None:
    tools.append(Tool(
        name="TesseraMiner",
        func=tool_guard("TesseraMiner", search_knowledge_base),
        description="Search scientific publications in TesseraStore."
    ))
tools.append(Tool(
    name="DataFinder",
    func=tool_guard("DataFinder", search_osdr_database),
    description="Find raw datasets in NASA OSDR by keyword or project name."
))

# --- AGENT ---
@st.cache_resource(show_spinner=False)
def build_agent(_tools):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)
    prompt_template = """
You are a specialized AI assistant for NASA called Biotessera.
Answer questions about space biology based ONLY on your tools.
Always cite sources. For TesseraMiner, cite 'Source' title. For DataFinder, provide 'Title' and 'Link'.
If you have used a tool 2 times and the result is similar, STOP using tools and produce the Final Answer with what you have.

You have access to the following tools:
{tools}

Use this format:
Question: the user's input question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result
... (repeat as needed)
Thought: final reasoning
Final Answer: well-formatted answer with citations.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
    agent_prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, _tools, agent_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=_tools,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=4,
        early_stopping_method="generate",
    )
    return executor

agent_executor = build_agent(tools)

# --- UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask your question...")
if prompt:
    if st.session_state.last_prompt == prompt:
        st.stop()
    st.session_state.last_prompt = prompt

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    tool_calls["TesseraMiner"] = 0
    tool_calls["DataFinder"] = 0

    with st.chat_message("assistant"):
        steps = []
        with st.spinner("Biotessera is thinking..."):
            try:
                resp = agent_executor.invoke({"input": prompt})
                text = resp.get("output", "Sorry, an error occurred.")
                steps = resp.get("intermediate_steps", [])
            except Exception as e:
                text = f"❌ Agent error: {e}"
                steps = []
            st.markdown(text)

            if steps:
                tools_used = [getattr(a, "tool", "unknown") for (a, _) in steps]
                if tools_used:
                    used_str = ", ".join(tools_used)
                    st.info(f"🧠 Tools used in reasoning: {used_str}")
                else:
                    st.info("🧠 Tools used in reasoning: —")

                if "DataFinder" not in tools_used:
                    st.warning("DataFinder was not called in this run. (Model may have decided that there were enough publications.)")

                import re
                def sanitize(s: str) -> str:
                    if not isinstance(s, str):
                        return str(s)
                    return re.sub(r"(?im)^\s*thought\s*:.*$", "[redacted thought]", s)

                with st.expander("🧭 Agent Workflow (tools & observations)"):
                    for i, (action, observation) in enumerate(steps, 1):
                        tool_name = getattr(action, "tool", "unknown")
                        tool_input = sanitize(getattr(action, "tool_input", ""))
                        st.markdown(f"**Step {i}: {tool_name}**")
                        if tool_input:
                            st.code(str(tool_input), language="text")
                        obs_str = observation if isinstance(observation, str) else str(observation)
                        st.text(obs_str)
                        st.divider()

            st.session_state.messages.append({"role": "assistant", "content": text})
