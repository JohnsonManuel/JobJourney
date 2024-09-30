import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def create_faiss_vectorstores(directory_path, faiss_index_path=None):
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-v4")
    
    # Initialize an empty list to collect documents with metadata
    all_documents = []
    
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return None
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)

            try:
                # Load the PDF and split it into pages
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()

                # Filter out empty pages
                documents = [page for page in pages if page.page_content]

                # Initialize a text splitter with chunk size and overlap
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = [text for text in text_splitter.split_documents(documents) if text.page_content.strip()]

                # Add metadata (filename and page number)
                for i, text in enumerate(texts):
                    text.metadata['source'] = filename
                    text.metadata['page'] = i + 1
                    all_documents.append(text)
            
            except Exception as e:
                print(f"Error processing PDF {filename}: {e}")
                continue

    # Check if any texts were collected
    if not all_documents:
        print("No PDF files found in the directory or PDFs are empty.")
        return None
    
    # Create or load the FAISS vectorstore
    try:
        if faiss_index_path and os.path.exists(faiss_index_path):
            vectorstore = FAISS.load_local(faiss_index_path, embedding_model)
        else:
            vectorstore = FAISS.from_documents(all_documents, embedding_model)

        # Persist the vectorstore if path provided
        if faiss_index_path:
            vectorstore.save_local(faiss_index_path)

        return vectorstore
    
    except Exception as e:
        print(f"Error creating or loading FAISS vectorstore: {e}")
        return None


# def create_faiss_vectorstores(directory_path):
#     # Initialize the embedding model
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
#     # Initialize an empty list to collect documents
#     all_texts = []
    
#     # Check if the directory exists
#     if os.path.isdir(directory_path):
#         # Iterate over all files in the directory
#         for filename in os.listdir(directory_path):
#             # Filter to only PDF files
#             if filename.endswith(".pdf"):
#                 file_path = os.path.join(directory_path, filename)

#                 # Load the PDF and split it into pages
#                 loader = PyPDFLoader(file_path)
#                 pages = loader.load_and_split()

#                 # Filter out empty pages
#                 documents = [page for page in pages if page.page_content]

#                 # Initialize a text splitter
#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 texts = text_splitter.split_documents(documents)

#                 # Collect texts from all PDFs
#                 all_texts.extend(texts)
        
#         # Check if any texts were collected
#         if all_texts:
#             # Create the vectorstore from all collected texts
#             vectorstore = FAISS.from_documents(all_texts, embedding_model)
#             return vectorstore
#         else:
#             print("No PDF files found in the directory or PDFs are empty.")
#             return None
#     else:
#         print(f"Directory {directory_path} does not exist.")
#         return None

