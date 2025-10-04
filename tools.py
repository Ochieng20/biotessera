# tools.py
from langchain_core.tools import Tool

# --- Tool 1: TesseraMiner ---
def search_knowledge_base(query: str, vector_db) -> str:
    """Searches the TesseraStore for relevant and diverse text chunks using MMR."""
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
    results = retriever.invoke(query)
    return "\n---\n".join([f"Source: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}" for doc in results])

def get_tools(vector_db):
    """Initializes and returns a list of all agent tools."""
    tessera_miner_tool = Tool(
        name="TesseraMiner",
        func=lambda query: search_knowledge_base(query, vector_db),
        description="Use this tool to search scientific publications for information about space biology, microgravity effects, and experimental results. The query should be a detailed question in English."
    )
    return [tessera_miner_tool]