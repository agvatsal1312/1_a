FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and source code
COPY model.pt .
COPY main2.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the entrypoint script
COPY entrypoint.py .

# Run the entrypoint
CMD ["python", "entrypoint.py"] 