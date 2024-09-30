import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

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

    def generate_response(self, resume_text, job_description, instruction):
        """Generate a response from the Groq LLM using a structured prompt."""
        # Define the prompt structure using PromptTemplate
        prompt_template = PromptTemplate.from_template(
            """
            ### RESUME:
            {resume_text}

            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            {instruction}

            ### INSTRUCTION FOR AI:
            Return your answer in JSON format with keys `improvements`, `missing_skills`, and `overall_assessment`. Ensure the response is a valid JSON object without preamble.
            """
        )

        # Create the chain with the Groq LLM
        chain_check = prompt_template | self.llm
        
        # Invoke the LLM with the prompt
        try:
            res = chain_check.invoke({
                "resume_text": resume_text,
                "job_description": job_description,
                "instruction": instruction
            })
            # Parse the response using JsonOutputParser
            json_parser = JsonOutputParser()
            result = json_parser.parse(res.content)
        except OutputParserException as e:
            raise OutputParserException(f"Unable to parse response: {e}")
        return result

#     def optimize_resume(self, job_description, current_resume):
#         """Optimize a resume based on the job description."""
#         context = self.query_rag(job_description)

#         # Define the instruction for optimizing the resume
#         instruction = f"Analyze the resume in the context of the job description provided. Identify areas where the resume can be improved to better match the job requirements."
        
#         return self.generate_response(current_resume, job_description, instruction)

#     def write_cover_letter(self, job_description):
#         """Write a cover letter based on the job description."""
#         context = self.query_rag(job_description)

#         # Define the instruction for writing the cover letter
#         instruction = "Write a cover letter for the job description provided. Ensure that the cover letter highlights relevant experience and skills."
        
#         return self.generate_response("N/A", job_description, instruction)

#     def write_cold_email(self, job_description):
#         """Write a cold email based on the job description."""
#         context = self.query_rag(job_description)

#         # Define the instruction for writing the cold email
#         instruction = "Write a professional cold email introducing the applicant and their interest in the job described."
        
#         return self.generate_response("N/A", job_description, instruction)

#     def chat_agent(self, query):
#         """AI Assistant to guide the job seeker in career advice."""
#         context = self.query_rag(query)

#         # Define the instruction for career guidance
#         instruction = f"Provide guidance on the next steps for the job seeker based on their query: '{query}'. Suggest specific actions to improve their chances of landing a job."

#         return self.generate_response("N/A", "N/A", instruction)

# # Example Usage
# if __name__ == "__main__":
#     joblander = Joblander()
#     joblander.initialize_knowledge_base("result.txt")
    
#     job_description = "Software Engineer position at XYZ Corp focusing on Python and machine learning."
#     current_resume = "Experienced software engineer with skills in Python, C++, and cloud computing."
    
#     optimized_resume = joblander.optimize_resume(job_description, current_resume)
#     cover_letter = joblander.write_cover_letter(job_description)
#     cold_email = joblander.write_cold_email(job_description)
    
#     print("Optimized Resume:\n", optimized_resume)
#     print("\nCover Letter:\n", cover_letter)
#     print("\nCold Email:\n", cold_email)
    
#     # AI Assistant Query Example
#     query = "I want to become a machine learning engineer, what should I do?"
#     assistant_response = joblander.chat_agent(query)
#     print("\nAI Assistant Response:\n", assistant_response)
