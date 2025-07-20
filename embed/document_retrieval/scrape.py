import pymupdf4llm
from llama_index.core.schema import Document
import json
import os
from pathlib import Path

current_directory = os.getcwd()

# Append '../documents' to the current working directory
folder_path = os.path.join(current_directory, 'embed', 'documents')

# Convert the relative path to an absolute path
folder_path = os.path.abspath(folder_path)
llama_reader = pymupdf4llm.LlamaMarkdownReader()

def save_llama_docs_to_json(llama_docs: list[Document], output_path: str):
    data = []
    for doc in llama_docs:
        data.append({
            "text": doc.text,
            "metadata": doc.metadata
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {output_path}")

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        
        try:
            print(f"Processing {file_path}...")
            llama_docs = llama_reader.load_data(file_path)
            json_path = Path(file_path).with_suffix('.json')
            save_llama_docs_to_json(llama_docs, json_path)
            
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

