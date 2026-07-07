FROM python:3.11-slim

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy dependency requirements first to leverage Docker layer caching
COPY --chown=user:user backend/requirements.txt ./backend/
RUN pip install --no-cache-dir --user -r backend/requirements.txt

# Copy the entire workspace contents
COPY --chown=user:user . .

# Set working directory to backend so that app.py runs with correct local imports/paths
WORKDIR $HOME/app/backend

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "app.py"]
