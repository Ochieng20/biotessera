import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# --- SETTINGS ---
load_dotenv()
PERSIST_DIR = 'tesserastore_db' # Folder with persisted vector DB

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("Error: Environment variable GOOGLE_API_KEY not found.")
    exit()

def check_vector_store():
    """Load an existing vector store and run a test similarity search."""

    print("Initializing embedding model...")
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print(f"Loading vector database from '{PERSIST_DIR}'...")
    # Load the existing DB from disk
    vector_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function
    )
    
    # --- TEST QUERY ---
    query = "What is the effect of microgravity on bone loss?"
    print(f"\nPerforming similarity search for query: '{query}'")
    
    # Perform semantic search (by default returns top 4 most relevant documents)
    results = vector_db.similarity_search(query)
    
    print(f"\nFound {len(results)} relevant chunks:")
    print("-" * 50)

    # Print results
    for i, doc in enumerate(results):
        print(f"--- RESULT {i+1} ---")
        print(f"Source Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Source Authors: {doc.metadata.get('authors', 'N/A')}")
        print("\nRelevant Chunk:")
        print(doc.page_content)
        print("-" * 50)

if __name__ == '__main__':
    check_vector_store()