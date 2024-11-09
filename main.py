import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
from io import BytesIO
import fitz  # PyMuPDF

# Set device based on GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load embeddings with a smaller model and run on CPU
embedding_model = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})

# Set up text generation model with PyTorch-compatible pipeline
model_name = "google/flan-t5-small"  # Or use a smaller model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Create a text generation pipeline
generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    model_kwargs={"max_length": 256, "temperature": 0.7}
)

llm = HuggingFacePipeline(pipeline=generator)

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

            # Generate response
            try:
                response = generator(prompt, max_new_tokens=50, num_return_sequences=1)[0]["generated_text"]
                st.write(response)
            except torch.cuda.OutOfMemoryError:
                st.error("Out of memory. Try using a smaller model or fewer documents.")

if __name__ == "__main__":
    main()
