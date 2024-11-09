# Use the official Python image as a base
FROM python:3.8-slim

# Set environment variables for Python and paths
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies (including Python and pip)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install PyTorch without GPU support
RUN pip install torch==2.0.1

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the current directory into the container
COPY . /app/

# Install the required Python dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Start the Streamlit app when the container starts
CMD ["streamlit", "run", "app.py"]
