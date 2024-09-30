# imports
from langchain_community.vectorstores import FAISS
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
import os
from langchain_community.embeddings import HuggingFaceEmbeddings  # Or OpenAIEmbeddings or any other embedding model
from fpdf import FPDF



groq_api_key = os.environ["GROQ_API_KEY"]

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",  # Adjust this model name as per Groq's API documentation.
    temperature=0,
    max_retries=2,
    api_key=groq_api_key
)

def load_faiss_store(store_name, embeddings):
    """Loads FAISS store based on the provided store name."""
    try:
        store = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
        print(f"{store_name} FAISS store loaded successfully.")
        return store
    except Exception as e:
        print(f"Error loading {store_name} FAISS store: {e}")
        return None

def load_env(username, url):
    """Load all necessary FAISS stores and job details."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/msmarco-distilbert-base-v4")
    
    pdf_store = load_faiss_store(f'{username}_pdf_faiss_store', embeddings)
    git_store = load_faiss_store(f'{username}_git_faiss_store', embeddings)
    personal_details_store = load_faiss_store(f'{username}_person_store', embeddings)
    job_details = web_scrapper(url)
    
    return pdf_store, git_store, personal_details_store, job_details

def generate_output(prompt):
    try:
        response = llm.invoke(input=prompt)
        if hasattr(response, "content"):
            return response.content
        else:
            return str(response)
    except Exception as e:
        print(f"Error interacting with LLM: {e}")
        return None

def automatic_suggestion(username, url):
    pdf_store, git_store, personal_details_store, job_details = load_env(username, url)
    
    if not pdf_store or not git_store or not personal_details_store or not job_details:
        return None

    try:
        resume = pdf_store.similarity_search("resume")[0].page_content
        personal_details = personal_details_store.similarity_search("personal details")[0].page_content
        github_summary = git_store.similarity_search("GitHub projects")[0].page_content
    except Exception as e:
        print(f"Error extracting data from FAISS stores: {e}")
        return None

    prompt = f"""
    You are an AI assistant. Based on the following data:

    Resume Information: {resume}
    Personal Details: {personal_details}
    GitHub Summary: {github_summary}
    Job Summary: {job_details}

    Please provide suggestions on how the user can improve their chances of landing a job, including improvements to their resume, additional skills to focus on, or projects they could work on and provide the text in the resume format and make the heading bold and follow the resume standat.
    """
    return generate_output(prompt)

def write_cold_email(username, url):
    pdf_store, git_store, personal_details_store, job_details = load_env(username, url)
    
    if not pdf_store or not personal_details_store or not job_details:
        return None

    resume = pdf_store.similarity_search("resume")[0].page_content
    personal_details = personal_details_store.similarity_search("personal details")[0].page_content

    prompt = f"""
    You are an AI assistant specializing in helping job seekers. Based on the following personal details:

    Personal Details: {personal_details}
    Resume Details: {resume}

    Write a professional cold email tailored for the job described below.

    Job Description: {job_details}
    The email should be concise and focus on the applicant's strengths.
    """
    return generate_output(prompt)

def create_roadmap(username, url):
    pdf_store, git_store, personal_details_store, job_details = load_env(username, url)
    
    if not personal_details_store or not git_store or not job_details:
        return None

    personal_details = personal_details_store.similarity_search("personal details")[0].page_content
    github_summary = git_store.similarity_search("GitHub projects")[0].page_content

    prompt = f"""
    You are an AI assistant. Create a learning roadmap for this user based on the following details:

    Personal Details: {personal_details}
    GitHub Summary: {github_summary}
    Job Description: {job_details}

    The roadmap should guide the user towards their learning goals and career objectives.
    """
    return generate_output(prompt)

def write_cover_letter(username, url):
    pdf_store, git_store, personal_details_store, job_details = load_env(username, url)
    
    if not pdf_store or not personal_details_store or not job_details:
        return None

    cover_letter = pdf_store.similarity_search("cover letter")[0].page_content
    personal_details = personal_details_store.similarity_search("personal details")[0].page_content

    prompt = f"""
    You are an AI that specializes in writing cover letters. Write a persuasive cover letter for the job described below.

    Job Description: {job_details}
    Personal Details: {personal_details}
    Cover Letter Draft: {cover_letter}

    Ensure that it highlights the applicant's relevant skills and experience effectively.
    """
    return generate_output(prompt)

def write_resume(username, url):
    pdf_store, git_store, personal_details_store, job_details = load_env(username, url)
    
    if not pdf_store or not git_store or not personal_details_store or not job_details:
        return None

    resume = pdf_store.similarity_search("resume")[0].page_content
    personal_details = personal_details_store.similarity_search("personal details")[0].page_content
    github_summary = git_store.similarity_search("GitHub projects")[0].page_content

    prompt = f"""
    You are an AI that optimizes resumes. Please optimize the following resume to better align with the job description.

    Job Description: {job_details}
    Personal Details: {personal_details}
    GitHub Summary: {github_summary}
    Current Resume: {resume}

    The optimized resume should highlight the applicant's key strengths and align with the job requirements.
    """
    return generate_output(prompt)

def web_scrapper(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"Error scraping the webpage: {e}")
        return None

def text_to_pdf(text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    sanitized_text = text.encode('ascii', 'ignore').decode()
    pdf.multi_cell(0, 10, sanitized_text)
    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output
