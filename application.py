# from Ats import ATSResumeChecker
from pdf_scrapper import create_faiss_vectorstores
from git_scrapper import process_githublink
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import copy
from langchain.schema import Document

from rag_functions import load_env


def executeMainProcess(link, directory_path ,url, username = "user_1"):
    # Step 1: Create the FAISS vector stores using Langchain
    pdf_store = create_faiss_vectorstores(directory_path)  # This returns a FAISS VectorStore object
    git_store = process_githublink(link)  # This also returns a FAISS VectorStore object
    person_store = loadlocal()
    # Save the vector stores.
    pdf_store.save_local(f'{username}_pdf_faiss_store')
    git_store.save_local(f'{username}_git_faiss_store')
    person_store.save_local(f'{username}_person_store')
    
    # load_env(username ,url)
   

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def loadlocal():
    # Load local data from local_data.txt
    file_path = 'local_data.txt'
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"The file {file_path} was not found.")
        return None

    try:
        with open(file_path, 'r') as file:
            data = file.readlines()
        
        # Ensure there is data to process
        if not data:
            print("The file is empty.")
            return None

        print(data)

        # Process the data into documents for embedding
        documents = [Document(page_content=line.strip()) for line in data if line.strip()]

        # Check if documents are created
        if not documents:
            print("No valid documents found to embed.")
            return None

        # Initialize HuggingFace embeddings model
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/msmarco-distilbert-base-v4')

        # Create FAISS index from documents using the embeddings
        faiss_index = FAISS.from_documents(documents, embeddings)

        return faiss_index

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

