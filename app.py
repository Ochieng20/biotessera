import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

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

# --- FUNCTIONS ---
@st.cache_resource
def load_knowledge_base():
    """Downloads the TesseraStore once and caches the result."""
    print("🔄 Loading TesseraStore for the first time...")
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embedding_function
        )
        print("✅ TesseraStore loaded successfully!")
        return vector_db
    except Exception as e:
        print(f"❌ Error loading TesseraStore: {e}")
        return None
    
@st.cache_resource
def initialize_agent(_vector_db):
    """Initializes and caches the AI ​​agent."""
    print("🚀 Initializing Biotessera Agent...")
    try:
        GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
        genai.configure(api_key=GOOGLE_API_KEY)
    except KeyError:
        print("❌ Error: GOOGLE_API_KEY not found.")
        return None
    
    tools = get_tools(_vector_db)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

    prompt_template = """
    You are a specialized AI assistant for NASA called Biotessera.
    Your task is to answer questions about space biology based *only* on the information provided by your tools.
    When you find information, you MUST cite your sources. For TesseraMiner, cite the 'Source' title. For DataFinder, provide the 'Title' and 'Link'.

    You have access to the following tools:
    {tools}

    Use the following format. Do not write the final answer in the Thought block.

    Question: the user's input question
    Thought: I need to break down the user's question and decide which tools to use. I will execute all necessary tool calls before forming a final answer.
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I have gathered all necessary information and I am ready to give the final answer.
    Final Answer: [The final, comprehensive, markdown-formatted answer to the original question, with citations.]

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
    agent_prompt = PromptTemplate.from_template(prompt_template)

    biotessera_agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=biotessera_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    print("✅ Agent Executor created. Biotessera is ready!")
    return agent_executor

# --- DATA PATHS ---
ROOT_DIR = Path(__file__).parent
PERSIST_DIR = ROOT_DIR / 'tesserastore_db'

# --- MAIN APP ---
st.set_page_config(page_title="Biotessera", page_icon="🧩")
st.title("🧩 Biotessera")
st.caption("An AI-powered knowledge engine for NASA's space biology research.")

vector_db = load_knowledge_base()
agent_executor = initialize_agent(vector_db)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about space biology..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Biotessera is thinking..."):
            if agent_executor:
                agent_executor.return_intermediate_steps = True
                
                response = agent_executor.invoke({"input": prompt})
                response_text = response.get('output', "Sorry, I encountered an error.")
                
                st.markdown(response_text)
                
                if 'intermediate_steps' in response and response['intermediate_steps']:
                    with st.expander("🤖 Show Agent's Thought Process"):
                        for step in response['intermediate_steps']:
                            action = step[0]
                            observation = step[1]
                            st.markdown(f"**Action:** `{action.tool}`")
                            st.markdown(f"**Action Input:** `{action.tool_input}`")
                            st.markdown(f"**Observation:**")
                            st.text(observation)
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.error("Agent could not be initialized.")