import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_groq import ChatGroq  # Import Groq client for handling the chat completions

# Avoid OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Joblander:
    def __init__(self):
        # Initialize the LLM using Groq
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-70b-versatile"
        )

        # Initialize the HuggingFace embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Initialize FAISS index for similarity search
        self.dimension = 384  # Assuming the HuggingFace model has a 384-dimensional output
        self.index = faiss.IndexFlatL2(self.dimension)

    def embed_text(self, text):
        """Embed text using HuggingFace transformers."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def read_knowledge_base(self, file_path):
        """Read and return lines from the knowledge base text file."""
        with open(file_path, 'r') as file:
            data = file.readlines()
        return data

    def initialize_knowledge_base(self, knowledge_base_file):
        """Initialize FAISS index with embeddings from the knowledge base."""
        knowledge_base = self.read_knowledge_base(knowledge_base_file)
        
        # Embed the knowledge base into a list of vectors
        knowledge_base_embeddings = [self.embed_text(text) for text in knowledge_base]
        
        # Convert the list of embeddings into a 2D numpy array
        knowledge_base_embeddings = np.vstack(knowledge_base_embeddings)
        
        # Add the embeddings to the FAISS index
        self.index.add(knowledge_base_embeddings)
        self.knowledge_base = knowledge_base

    def query_rag(self, input_text):
        """Perform a similarity search on the FAISS index to get relevant context."""
        input_embedding = self.embed_text(input_text)
        D, I = self.index.search(np.array(input_embedding), k=5)
        retrieved_texts = [self.knowledge_base[i] for i in I[0]]
        context = "\n".join(retrieved_texts)
        return context

    def generate_response(self, messages, model="llama3-8b-8192", temperature=0.5, max_tokens=1024):
        """Generate a response from the Groq LLM using streaming."""
        stream = self.llm(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            stop=None,
            stream=True
        )
        
        # Collect the generated output
        result = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:  # Avoid concatenating None
                result += content

        return result

    def optimize_resume(self, job_description, current_resume):
        """Optimize a resume based on the job description."""
        context = self.query_rag(job_description)
        
        # Properly structured messages for Groq
        messages = [
            {"role": "system", "content": "You are a resume optimization assistant."},
            {"role": "user", "content": f"Context: {context}\n\nOptimize this resume:\n{current_resume}\n\nJob Description: {job_description}\nOptimized Resume:"}
        ]
        
        return self.generate_response(messages)

    def write_cover_letter(self, job_description):
        """Write a cover letter based on the job description."""
        context = self.query_rag(job_description)
        
        # Properly structured messages for Groq
        messages = [
            {"role": "system", "content": "You are a cover letter assistant."},
            {"role": "user", "content": f"Context: {context}\n\nWrite a cover letter for the following job description:\n{job_description}\n\nCover Letter:"}
        ]
        
        return self.generate_response(messages)

    def write_cold_email(self, job_description):
        """Write a cold email based on the job description."""
        context = self.query_rag(job_description)
        
        # Properly structured messages for Groq
        messages = [
            {"role": "system", "content": "You are a cold email assistant."},
            {"role": "user", "content": f"Context: {context}\n\nWrite a professional cold email for the following job description:\n{job_description}\n\nCold Email:"}
        ]
        
        return self.generate_response(messages)

    def chat_agent(self, query):
        """AI Assistant to guide the job seeker in career advice."""
        context = self.query_rag(query)
        
        # Properly structured messages for Groq
        messages = [
            {"role": "system", "content": "You are a career guidance assistant."},
            {"role": "user", "content": f"Context: {context}\n\nBased on the query '{query}', provide guidance to help the job seeker improve their chances of getting a job. Suggest courses, projects, and steps they should take."}
        ]
        
        return self.generate_response(messages)

# Example Usage
if __name__ == "__main__":
    joblander = Joblander()
    joblander.initialize_knowledge_base("result.txt")
    
    job_description = "Software Engineer position at XYZ Corp focusing on Python and machine learning."
    current_resume = "Experienced software engineer with skills in Python, C++, and cloud computing."
    
    optimized_resume = joblander.optimize_resume(job_description, current_resume)
    cover_letter = joblander.write_cover_letter(job_description)
    cold_email = joblander.write_cold_email(job_description)
    
    print("Optimized Resume:\n", optimized_resume)
    print("\nCover Letter:\n", cover_letter)
    print("\nCold Email:\n", cold_email)
    
    # AI Assistant Query Example
    query = "I want to become a machine learning engineer, what should I do?"
    assistant_response = joblander.chat_agent(query)
    print("\nAI Assistant Response:\n", assistant_response)
