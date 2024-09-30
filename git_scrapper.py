import os
from github import Github
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

github_key = os.environ.get('GITHUB')

g = Github(github_key)


groq_api_key = os.environ["GROQ_API_KEY"]

# Initialize the Groq LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",  # Adjust this model name as per Groq's API documentation.
    temperature=0,
    max_retries=2,
    api_key=groq_api_key
)

# Function to fetch repository READMEs
def fetch_repositories(link):
    repo_owner = link.split('/')[-1]
    repo_readme_list = []
    named_user = g.get_user(repo_owner)
    repos = named_user.get_repos()
    
    MINIMUM_LENGTH = 50  # Set the minimum length for the README content (in characters or words)
    
    for repo in repos:
        try:
            curr_repo = g.get_repo(f"{repo_owner}/{repo.name}")
            read_me = curr_repo.get_readme()
            readme_content = read_me.decoded_content.decode('utf-8')  # Handle encoding errors gracefully
            
            # Check if the README content is long enough
            if readme_content and len(readme_content.strip()) >= MINIMUM_LENGTH:
                repo_readme_list.append({
                    'repo_name': repo.name,
                    'readme_content': readme_content,
                    'repo_owner': repo_owner
                })
            else:
                print(f"Skipping repository {repo.name} due to small README content.")
        
        except Exception as e:
            continue  # Optionally, log the error if needed
            # print(f"Error fetching README for repository {repo.name}: {e}")
    
    return repo_readme_list

# def fetch_repositories(link):
#     repo_owner = link.split('/')[-1]
#     repo_readme_list = []
#     # Fetch the user's repositories
#     named_user = g.get_user(repo_owner)
#     repos = named_user.get_repos()
#     for repo in repos:
#         try:
#             # Fetch the README content for each repository
#             curr_repo = g.get_repo(f"{repo_owner}/{repo.name}")
#             read_me = curr_repo.get_readme()
#             readme_content = read_me.decoded_content.decode('utf-8')
#         except Exception as e:
#             readme_content = None
#         # Add the repository name and README content to the list
#         if readme_content:
#             repo_readme_list.append({
#                 'repo_name': repo.name,
#                 'readme_content': readme_content,
#                 'repo_owner': repo_owner
#             })
#     return repo_readme_list


# Process repositories to get READMEs
# repo_list = process_repositories(repo_link)


def summarise_repositories(repo_list):
    # Define the summarization prompt
    summary_prompt = PromptTemplate(
        input_variables=["document"],
        template="Create a concise summary of the following project, including key information such as the main goal, core technologies used, and key accomplishments. Limit the summary to 50 words: {document}"
    )
    
    # Chain the prompt and the LLM using the `|` operator
    summarization_chain = summary_prompt | llm
    
    # Summarize each repository's README content
    for repo in repo_list:
        if repo['readme_content']  :
            try:
                # Pass the README content directly as a string to the `.run()` method
                summary = summarization_chain.invoke({"document": repo['readme_content']}).content
                repo['readme_content'] = summary  # Replace the content with the summary
            except Exception as e:
                print(f"Error summarizing repository {repo['repo_name']}: {e}")
    
    return repo_list


def embed_repositories(repo_list):
    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-v4")
    
    # List to store documents for Langchain's FAISS integration
    documents = []

    # Loop through the list of repositories and embed the README content
    for repo in repo_list:
        if 'readme_content' in repo and repo['readme_content']:
            try:
                # Create a Document object with metadata (repo_name) and content (readme_content)
                document = Document(
                    page_content=repo['readme_content'] +' ' + repo['repo_name'] + ' '+ repo["repo_owner"]
                    # metadata={'repo_name': repo['repo_name'] , 'repo_owner': repo['repo_owner']}
                )
                documents.append(document)
            except Exception as e:
                print(f"Error processing repository {repo['repo_name']}: {e}")
    
    # Embed the documents and create a FAISS vector store
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    # Return the unified FAISS vector store containing both embeddings and metadata
    return vector_store
# Create the LLM chain for summarization


def process_githublink(link):
    repo_list = fetch_repositories(link)
    summarised_repo_list = summarise_repositories(repo_list)
    return embed_repositories(summarised_repo_list)                




process_githublink("https://github.com/KiranVSurendran")

