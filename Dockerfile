FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copy the processing script
COPY process_documents.py .

# Run the document analyzer
CMD ["python", "process_documents.py"]
