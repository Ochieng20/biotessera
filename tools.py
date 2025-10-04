# tools.py
import requests
from bs4 import BeautifulSoup
import urllib.parse
from langchain_core.tools import Tool

# --- Tool 1: TesseraMiner ---
def search_knowledge_base(query: str, vector_db) -> str:
    """Searches the TesseraStore for relevant and diverse text chunks using MMR."""
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
    results = retriever.invoke(query)
    return "\n---\n".join([f"Source: {doc.metadata.get('title', 'N/A')}\nContent: {doc.page_content}" for doc in results])

# --- Tool 2: DataFinder ---
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
        results = soup.find_all('div', class_='search-result-row-inner')
        if not results:
            return "No datasets found in OSDR for this query."
        found_datasets = []
        for res in results[:3]:
            title_tag = res.find('a')
            title = title_tag.text.strip() if title_tag else "No title found"
            link = "https://osdr.nasa.gov" + title_tag['href'] if title_tag and title_tag.has_attr('href') else "No link found"
            found_datasets.append(f"Title: {title}\nLink: {link}")
        return "\n---\n".join(found_datasets)
    except requests.exceptions.RequestException as e:
        return f"Error connecting to OSDR: {e}"
    except Exception as e:
        return f"An error occurred while searching OSDR: {e}"

def get_tools(vector_db):
    """Initializes and returns a list of all agent tools."""
    tessera_miner_tool = Tool(
        name="TesseraMiner",
        func=lambda query: search_knowledge_base(query, vector_db),
        description="Use this tool to search scientific publications for information about space biology, microgravity effects, and experimental results. The query should be a detailed question in English."
    )
    data_finder_tool = Tool(
        name="DataFinder",
        func=search_osdr_database,
        description="Use this tool to find raw datasets in the NASA Open Science Data Repository (OSDR). The query should be a simple keyword or project name, for example 'Bion-M1' or 'osteoclast'."
    )
    return [tessera_miner_tool, data_finder_tool]