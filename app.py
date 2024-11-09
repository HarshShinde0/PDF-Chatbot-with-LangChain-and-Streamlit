import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.schema import Document
import requests
from io import BytesIO
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Set device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load environment variables from .env file
load_dotenv()

# Hugging Face API token should now be loaded from the .env file
# Explicitly set the Hugging Face API token from the environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")

# Load embeddings with Hugging Face API
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)  # Removed api_key parameter

# Set up the text generation model using Hugging Face Hub
model_name = "google/flan-t5-small"  # Use a smaller model to reduce response time and cost
llm = HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), model_kwargs={"max_length": 256, "temperature": 0.7})

# Streamlit interface
def main():
    st.title("Chat with Multiple PDFs")
    st.write("Upload PDF files and chat with them.")

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        # Load PDF documents
        documents = []
        for uploaded_file in uploaded_files:
            pdf_content = BytesIO(uploaded_file.read())
            doc = fitz.open(stream=pdf_content, filetype="pdf")  # Open PDF with PyMuPDF
            text = ""
            for page in doc:
                text += page.get_text()  # Extract text from each page
            doc.close()

            # Create Document instance with page content
            documents.append(Document(page_content=text, metadata={"file_name": uploaded_file.name}))

        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Embed document chunks into vector store
        vector_store = FAISS.from_documents(chunks, embeddings)

        # User query input
        st.write("You can now start chatting with your PDFs!")
        user_input = st.text_input("Ask a question:")

        if user_input:
            # Perform similarity search on the vector store
            docs = vector_store.similarity_search(user_input, k=3)

            # Concatenate retrieved docs into a single prompt
            prompt = "\n".join([doc.page_content for doc in docs]) + "\n\n" + user_input

            # Generate response using the Hugging Face API
            try:
                response = llm(prompt)
                st.write(response)
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to Hugging Face API: {e}")

if __name__ == "__main__":
    main()
