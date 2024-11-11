# [PDF Chatbot with LangChain, Hugging Face, and Streamlit](https://pdf-chatbot-with-langchain.streamlit.app/)


This project is a chatbot application that enables users to upload multiple PDF files and interact with their content through natural language queries. Using the LangChain library with Hugging Face embeddings and language models, the application extracts and vectorizes PDF content, allowing users to ask questions based on the uploaded documents. The project is deployed using Streamlit and Docker.

## Docker Image

This project’s Docker image is available on Docker Hub:

[![Docker Image](https://img.shields.io/badge/DockerHub-Image-blue?logo=docker&style=flat)](https://hub.docker.com/r/harshshinde/pdf-chat-app)

### Docker Commands

To pull and run the Docker container, use the following commands:

```bash
docker pull harshshinde/pdf-chat-app

docker run -d -p 8501:8501 harshshinde/pdf-chat-app
```

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [License](#license)

## Features

- **Upload Multiple PDFs**: Easily upload multiple PDF files for processing.
- **Chunked Text Splitting**: Text is split into manageable chunks for efficient vectorization and retrieval.
- **Document Embedding**: Use FAISS and Hugging Face sentence transformers to embed PDF content for vector-based similarity search.
- **Question Answering**: Leveraging a Hugging Face model, the app generates relevant responses based on the content of the uploaded PDFs.
- **User-Friendly Interface**: Built with Streamlit, providing a simple, interactive web interface.
- **Dockerized Deployment**: Easily deployable with Docker for consistent environment configuration.

## Getting Started

### Prerequisites

- **Docker**: Install [Docker](https://docs.docker.com/get-docker/).
- **Python 3.8 or Higher**: Required to run the application locally or configure the environment.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HarshShinde0/PDF-Chatbot-with-LangChain-and-Streamlit.git
   cd pdf-chatbot
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the root directory with your Hugging Face API token:
   ```plaintext
   HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
   ```

3. **Build and Run the Docker Container**:
   Build and run the application in a Docker container:
   ```bash
   docker-compose up --build
   ```

4. **Access the Application**:
   Open your web browser and go to `http://localhost:8501` to start interacting with the app.

## Project Structure

- `app.py`: Main Streamlit application that uses Hugging Face API for embeddings and text generation.
- `main.py`: Alternative app configuration, using a local PyTorch-compatible pipeline for text generation.
- `Dockerfile`: Docker configuration to create a containerized environment for the application.
- `docker-compose.yml`: Docker Compose setup to run the application, exposing the Streamlit port.
- `requirements.txt`: Lists all required Python libraries.

## File Descriptions

- **`app.py`**: The primary Streamlit app file, which includes:
  - PDF file upload handling
  - Text extraction from PDFs
  - Document chunking for efficient vectorization
  - Similarity search and question answering using Hugging Face models
- **`main.py`**: Contains an alternative setup using HuggingFacePipeline for text generation, which may be more suitable if using a GPU locally.
- **`Dockerfile`**: Specifies the Docker environment, installs required dependencies, and sets up the application.
- **`docker-compose.yml`**: Defines Docker services for running the Streamlit app, configures environment variables, and exposes port 8501.
- **`requirements.txt`**: Contains all Python dependencies necessary for the application.

## Usage

1. **Upload PDF Files**:  
   Click on the "Upload PDF Files" section to upload one or more PDFs. The uploaded PDFs will be loaded and preprocessed for interaction.

2. **Ask a Question**:  
   After uploading, type your question in the input box (e.g., "What is the main topic of this document?"). The app performs a similarity search within the PDF content and generates a response based on your question.

3. **Receive Responses**:  
   The application retrieves relevant chunks from the PDFs, generates a response using a language model, and displays the answer.

## Configuration Options

- **Embedding Model**: The default embedding model is `all-MiniLM-L6-v2`. You can configure a different embedding model from Hugging Face by modifying the model name in `app.py`.
- **Text Generation Model**: The `google/flan-t5-small` model is used for text generation, providing an efficient balance between response quality and resource usage. For larger documents or more complex questions, consider adjusting model size, though this may increase memory usage.

### Device Configuration:
- **GPU Support**: If CUDA is available on your device, the application will utilize it; otherwise, it defaults to CPU. Adjust device settings in `main.py` as needed.
- **Memory Optimization**: To avoid memory issues on limited-resource machines, try reducing the number of documents uploaded simultaneously or using a smaller language model.

## Troubleshooting

- **CUDA Out of Memory**: If you encounter a `CUDA OutOfMemoryError`, consider:
  - Using a smaller model (e.g., `google/flan-t5-small`).
  - Reducing the number of uploaded PDFs or the chunk size.
  - Running on CPU by setting `device = "cpu"` explicitly in `main.py`.
- **Connection Issues**: Ensure your Hugging Face API token in `.env` is valid and accessible.
- **Docker Errors**: If Docker fails to build, make sure all dependencies in `requirements.txt` are compatible with your environment.

## Dependencies

The project relies on the following libraries:

- `streamlit`: Provides the web interface for the application.
- `langchain`: Integrates language models and document handling.
- `faiss-cpu`: Enables fast similarity search and clustering.
- `pymupdf`: Extracts text from PDFs.
- `requests`: Handles API requests to Hugging Face.
- `transformers`: Provides models and tokenizers from Hugging Face.
- `sentence-transformers`: Facilitates sentence embedding for similarity search.
- `python-dotenv`: Manages environment variables from a `.env` file.
- `langchain-community`: Extends LangChain's functionality for specific integrations.

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

Feel free to contribute to the project by submitting pull requests or reporting issues. Happy chatting with your PDFs!

This README includes details for installation, setup, project structure, usage instructions, troubleshooting, and dependencies to help users fully understand and operate the PDF chatbot. Let me know if you’d like to add anything else!
