import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# --- SETTINGS ---
load_dotenv()

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Google AI API key is configured.")
except KeyError:
    print("❌ Error: Environment variable GOOGLE_API_KEY not found.")
    exit()

ROOT_DIR = Path(__file__).parent
PERSIST_DIR = ROOT_DIR / 'tesserastore_db'

print(f"🔄 Loading TesseraStore from: {PERSIST_DIR}")

# --- LOADING THE KNOWLEDGE BASE ---
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
    print("   Please ensure the 'tesserastore_db' folder exists and is not empty.")
    exit()

if __name__ == '__main__':
    # For testing
    print("\n--- Running a test query ---")
    test_query = "What is the effect of microgravity on bone loss?"
    results = vector_db.similarity_search(test_query)
    print(f"Found {len(results)} results for the test query.")
    print("First result snippet:")
    print(results[0].page_content[:500] + "...")
    print("--------------------------\n")