import os
import json
import re
import shutil
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

# Defining directories below.
DATA_DIRECTORY = "C://Users//edwar//Downloads//developer//DEV"
INDEX_DIRECTORY = "Index"  # The directory for storing index files.

# Creating the index directory, deletes the previous version if it already exists.
if os.path.exists(INDEX_DIRECTORY):
    shutil.rmtree(INDEX_DIRECTORY)
os.makedirs(INDEX_DIRECTORY)

# Initializing the necessary variables.
stemmer = PorterStemmer()
inverted_index = {}  # format - {"token": {"doc_id": freq}}
doc_count = 0

def extraction(html):
    soup = BeautifulSoup(html, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    text = ' '.join(text.split()).strip()
    return text

def tokenize(text: str) -> list:
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return tokens

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
            
            text = extraction(content)
            tokens = tokenize(text)
            stemmed_tokens = [stemmer.stem(token) for token in tokens if token.isalnum()]
            
            # Again, skipping if no valid tokens.
            if not stemmed_tokens:
                continue
            
            doc_id = doc_count
            doc_count += 1
            
            for token in stemmed_tokens:
                # Initializing a token if not already there.
                if token not in inverted_index:
                    inverted_index[token] = {}
                # Initializing a document if not already there.
                if doc_id not in inverted_index[token]:
                    inverted_index[token][doc_id] = 0
                # Incrementing frequency.
                inverted_index[token][doc_id] += 1

# Variables for storing partial index files and temporary size for memory contraints.
partial_index_files = []
temp_mem = 200000

def saving_partial_index(index, part_num):
    file_name = os.path.join(INDEX_DIRECTORY, f"index_part_{part_num}.json")
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)
    partial_index_files.append(file_name)

# Splitting index into multiple files.
part_num = 0
partial_index = {}
for token, tokens in inverted_index.items():
    partial_index[token] = tokens
    if len(partial_index) >= temp_mem:
        saving_partial_index(partial_index, part_num)
        partial_index = {}
        part_num += 1
if partial_index:
    saving_partial_index(partial_index, part_num)

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

# Saving the final_index.
with open(os.path.join(INDEX_DIRECTORY, "final_index.json"), "w", encoding="utf-8") as f:
    json.dump(final_index, f, indent=4)

# Generating the report.
index_size = os.path.getsize(os.path.join(INDEX_DIRECTORY, "final_index.json")) / 1024
unique_tokens = len(final_index)

data_report = {
    "Number of Documents": doc_count,
    "Number of Unique Tokens": unique_tokens,
    "Total Index Size (KB)": round(index_size, 2)
}

# Saving the report.
report_path = os.path.join(INDEX_DIRECTORY, "report.json")
with open(report_path, "w") as f:
    json.dump(data_report, f)

print(json.dumps(data_report))