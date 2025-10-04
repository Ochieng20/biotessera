import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
import re
import json
import traceback

# --- SETTINGS ---
CSV_FILE = 'SB_publication_PMC.csv'
OUTPUT_DIR = 'data'
# -----------------

def sanitize_filename(filename):
    """Remove characters that are invalid in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def download_and_extract_data():
    """Download HTML pages listed in a CSV, extract metadata and main text, and save as JSON files."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Folder '{OUTPUT_DIR}' created.")

    try:
        df = pd.read_csv(CSV_FILE)
        print(f"CSV file '{CSV_FILE}' successfully read. Found {len(df)} articles.")
    except FileNotFoundError:
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        return

    for index, row in df.iterrows():
        title_from_csv = row['Title']
        url = row['Link']
        
        safe_title = sanitize_filename(title_from_csv)[:100]
        filename = f"{str(index + 1).zfill(3)}_{safe_title}.json" # Save as .json
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            print(f"File '{filename}' already exists. Skipping.")
            continue

        print(f"[{index + 1}/{len(df)}] Processing: {title_from_csv}")

        try:
            headers = {'User-Agent': 'NASA Space Apps Participant Bot'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 1. Extract metadata from meta tags
            metadata = {}
            metadata['title'] = soup.find('meta', {'name': 'citation_title'})['content'] if soup.find('meta', {'name': 'citation_title'}) else 'N/A'
            metadata['authors'] = [tag['content'] for tag in soup.find_all('meta', {'name': 'citation_author'})]
            metadata['journal'] = soup.find('meta', {'name': 'citation_journal_title'})['content'] if soup.find('meta', {'name': 'citation_journal_title'}) else 'N/A'
            metadata['publication_date'] = soup.find('meta', {'name': 'citation_publication_date'})['content'] if soup.find('meta', {'name': 'citation_publication_date'}) else 'N/A'
            metadata['doi'] = soup.find('meta', {'name': 'citation_doi'})['content'] if soup.find('meta', {'name': 'citation_doi'}) else 'N/A'
            metadata['pmid'] = soup.find('meta', {'name': 'citation_pmid'})['content'] if soup.find('meta', {'name': 'citation_pmid'}) else 'N/A'
            
            # 2. Extract the main article text
            article_body = soup.find('section', class_='main-article-body')
            if article_body:
                text = article_body.get_text(separator=' ', strip=True)
            else:
                text = soup.body.get_text(separator=' ', strip=True)
                print("  > Warning: Main article container not found. Falling back to entire page text.")

            # 3. Combine into a single dictionary
            article_data = {
                'metadata': metadata,
                'full_text': text
            }

            # 4. Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, ensure_ascii=False, indent=4)
            print(f"  > Successfully saved to '{filename}'")

        except requests.exceptions.RequestException as e:
            # print(f"  > Error downloading article: {e}")
            print("\n" + "="*20 + " DOWNLOAD ERROR " + "="*20)
            print(f"CRITICAL DOWNLOAD ERROR on index {index}: {title_from_csv}")
            print(f"URL: {url}")
            print(f"Error: {e}")
            print("="*58 + "\n")
        except Exception as e:
            # print(f"  > An unexpected error occurred: {e}")
            print("\n" + "="*20 + " PROCESSING ERROR " + "="*20)
            print(f"CRITICAL PROCESSING ERROR on index {index}: {title_from_csv}")
            print(f"URL: {url}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error: {e}")
            traceback.print_exc() # Print a full error report
            print("="*58 + "\n")

        time.sleep(1)

    print("\nFinished debug run.")


if __name__ == '__main__':
    download_and_extract_data()