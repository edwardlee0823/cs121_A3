import os
import json
import re
import shutil
import math
import time
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

# Defining directories below.
DATA_DIRECTORY = "C://Users//edwar//Downloads//developer//DEV"
INDEX_DIRECTORY = "Index"  # The directory for storing index files.

DOC_MAP_FILE = os.path.join(INDEX_DIRECTORY, "doc_map.json")  # The file for mapping doc_id to URL.
FINAL_INDEX_FILE = os.path.join(INDEX_DIRECTORY, "final_index.json")
IDF_FILE = os.path.join(INDEX_DIRECTORY, "idf.json")
REPORT_FILE = os.path.join(INDEX_DIRECTORY, "report.json")

# Creating the index directory, deletes the previous version if it already exists.
if os.path.exists(INDEX_DIRECTORY):
    shutil.rmtree(INDEX_DIRECTORY)
os.makedirs(INDEX_DIRECTORY)

# Initializing the necessary variables.
stemmer = PorterStemmer()
inverted_index = {}  # format - {"token": {"doc_id": freq}}
doc_count = 0
doc_map = {}
temp_mem = 15000
idf_values = {}

def extraction(html):
    soup = BeautifulSoup(html, "html.parser")
    # While extraction, take care of HTML tags.
    HTML_tags = ["h1", "h2", "h3", "b", "strong"]
    important_text = []
    for tag in HTML_tags:
        for element in soup.find_all(tag):
            important_text.extend(element.get_text().split())
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    text = ' '.join(text.split()).strip()
    return text, important_text

def tokenize(text: str) -> list:
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return tokens

def saving_partial_index(index, part_num):
    file_name = os.path.join(INDEX_DIRECTORY, f"index_part_{part_num}.json")
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)
    return file_name

# Splitting index into multiple files.
partial_index_files = []
part_num = 0
partial_index = {}

# Processing documents.
for folder, _, files in os.walk(DATA_DIRECTORY):
    for file in files:
        file_path = os.path.join(folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            url = data.get("url", "")
            content = data.get("content", "")
            
            # Skipping if empty.
            if not content:
                continue
            text, important_text = extraction(content)
            tokens = tokenize(text)
            stemmed_tokens = [stemmer.stem(token) for token in tokens]

            doc_id = doc_count
            # Storing the mapping of doc_id to URL.
            doc_map[doc_id] = url
            doc_count += 1

            stemmed_important_text = set()
            for word in important_text:
                stemmed_word = stemmer.stem(word)
                stemmed_important_text.add(stemmed_word)

            for token in stemmed_tokens:
                # Increasing weight if important_text.
                weight = 2 if token in stemmed_important_text else 1
                # Initializing a token if not already there.
                if token not in inverted_index:
                    inverted_index[token] = {}
                # Initializing a document if not already there.
                if doc_id not in inverted_index[token]:
                    inverted_index[token][doc_id] = 0
                # Incrementing frequency.
                inverted_index[token][doc_id] += weight

        # Saving partial index every N documents.
        if doc_count % temp_mem == 0:
            partial_index_files.append(saving_partial_index(inverted_index, part_num))
            inverted_index.clear()
            part_num += 1

# Saving the remaining partial index.
if inverted_index:
    partial_index_files.append(saving_partial_index(inverted_index, part_num))

# Merging partial_index into the final_index.
final_index = {}
for file in partial_index_files:
    with open(file, "r", encoding="utf-8") as f:
        partial_index = json.load(f)
        for token, tokens in partial_index.items():
            # Initializing token if not already there.
            if token not in final_index:
                final_index[token] = {}
            for doc_id, freq in tokens.items():
                # Initializing document if not already there.
                if doc_id not in final_index[token]:
                    final_index[token][doc_id] = 0
                final_index[token][doc_id] += freq

# Computing and storing IDF values.
for token, posting in final_index.items():
    df = len(posting)
    idf_values[token] = math.log(doc_count / df) if df > 0 else 0

with open(FINAL_INDEX_FILE, "w", encoding="utf-8") as f:
    json.dump(final_index, f, indent=4)

with open(IDF_FILE, "w", encoding="utf-8") as f:
    json.dump(idf_values, f, indent=4)

with open(DOC_MAP_FILE, "w", encoding="utf-8") as f:
    json.dump(doc_map, f, indent=4)

# Generating the report.
index_size = os.path.getsize(FINAL_INDEX_FILE) / 1024
unique_tokens = len(final_index)

data_report = {
    "Number of Documents": doc_count,
    "Number of Unique Tokens": unique_tokens,
    "Total Index Size (KB)": round(index_size, 2)
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(data_report, f, indent=4)

print(json.dumps(data_report, indent=4))

def load_index():
    with open(FINAL_INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)
    with open(IDF_FILE, "r", encoding="utf-8") as f:
        idf_values = json.load(f)
    with open(DOC_MAP_FILE, "r", encoding="utf-8") as f:
        doc_map = {int(k): v for k, v in json.load(f).items()}
    
    return index, idf_values, doc_map

def search(query, index, idf_values, doc_map):
    tokens = [stemmer.stem(token) for token in tokenize(query)]
    relevant_docs = None
    
    for token in tokens:
        if token in index:
            doc_set = set(map(int, index[token].keys()))
            relevant_docs = doc_set if relevant_docs is None else relevant_docs & doc_set
        else:
            return [], doc_map

    # Computing TF-IDF scores
    scores = {}

    for doc_id in relevant_docs:
        total_score = 0
        sum_of_squares = 0
        for token in tokens:
            raw_tf = index[token].get(str(doc_id), 0)
            if raw_tf > 0:
                tf = 1 + math.log(raw_tf)
                idf = idf_values.get(token, 0)
                weighted_tf = tf * idf
                total_score += weighted_tf
                sum_of_squares += weighted_tf ** 2
        doc_length = math.sqrt(sum_of_squares)
        scores[doc_id] = total_score / doc_length if doc_length > 0 else 0

    ranked_docs = sorted(scores, key=scores.get, reverse=True)[:5]
    return ranked_docs, doc_map

def search_interface():
    print("\n  << Simple Search Engine >>  ")
    index, idf_values, doc_map = load_index()

    while True:
        query = input("Enter search query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
            
        if not query:
            print("ERROR: Query cannot be empty. Please enter a valid search term.")
            continue

        start_time = time.time()
        results, _ = search(query, index, idf_values, doc_map)
        elapsed_time = (time.time() - start_time) * 1000

        print(f"Search completed in {elapsed_time:.2f}ms")
        if not results:
            print("No results found.")
        else:
            print("\nTop 5 Search Results:")
            for rank, doc_id in enumerate(results, start=1):
                print(f"{rank}. {doc_map.get(doc_id, '[Missing URL]')}")

if __name__ == "__main__":
    search_interface()