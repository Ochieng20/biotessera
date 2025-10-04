import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm # for a nice progress bar
import google.generativeai as genai

# --- SETTINGS ---
load_dotenv()

DATA_DIR = 'data'
PERSIST_DIR = 'tesserastore_db'
CHUNK_SIZE = 1000 # Size of one chunk (in characters)
CHUNK_OVERLAP = 200 # How much chunks overlap to avoid losing context

try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("Error: Environment variable GOOGLE_API_KEY not found.")
    exit()

def load_documents_from_json(data_dir):
    """Load JSON files, handling errors, and convert them into a list of LangChain Documents."""
    documents = []
    print(f"Loading documents from directory '{data_dir}'...")
    
    # Use tqdm for a progress bar
    file_list = os.listdir(data_dir)
    for filename in tqdm(file_list, desc="Reading JSON files"):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Ensure the required keys exist in the JSON file
                if 'full_text' not in data or 'metadata' not in data:
                    print(f"\nWARNING: File {filename} is missing 'full_text' or 'metadata' key. Skipping file.")
                    continue

                # Convert the list of authors into a single string
                if 'authors' in data['metadata'] and isinstance(data['metadata']['authors'], list):
                    data['metadata']['authors'] = ', '.join(data['metadata']['authors'])

                # Create a Document object
                doc = Document(
                    page_content=data['full_text'],
                    metadata=data['metadata']
                )
                documents.append(doc)

            except json.JSONDecodeError:
                # This will catch files that are not valid JSON (e.g., empty or corrupted)
                print(f"\nERROR: Could not decode JSON from file: {filename}. The file might be corrupted. Skipping file.")
            except Exception as e:
                # This will catch any other unexpected errors
                print(f"\nUNEXPECTED ERROR processing file {filename}: {e}. Skipping file.")

    return documents

def main():
    """Main function to create and persist the TesseraStore."""
    
    # 1. Load documents
    documents = load_documents_from_json(DATA_DIR)

    # --- Diagnostics ---
    print("\n--- DIAGNOSTICS: Full file list ---")
    file_list = os.listdir(DATA_DIR)
    print(f"Found {len(file_list)} items in the directory:")
    print(file_list)
    print("--- END DIAGNOSTICS ---\n")

    if not documents:
        print("No documents found. Check the directory and file names.")
        return
    print(f"\nLoaded {len(documents)} documents.")

    # 2. Split texts into chunks ("tesserae")
    print("Splitting texts into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Texts split into {len(chunks)} chunks.")

    # 3. Initialize embedding model
    print("Initializing embedding model...")
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Create and persist the vector database
    print(f"Creating the vector database. This may take some time...")
    
    # - Takes each chunk
    # - Sends it to Google AI to obtain an embedding
    # - Stores text, metadata and vector in a local ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=PERSIST_DIR
    )

    print("\nDone!")
    print(f"The vector database 'TesseraStore' was created and saved in '{PERSIST_DIR}'.")


if __name__ == '__main__':
    main()