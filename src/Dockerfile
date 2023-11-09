# Use a base image with both C++ and Python
FROM python:3.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install C++ build tools and ffmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libsndfile1 \
    ffmpeg

# Clean up APT when done
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory to the root of the app
WORKDIR /app

# Copy the entire whisper directory
COPY whisper /app/whisper

# Set the working directory to the whisper directory
WORKDIR /app/whisper

# Download the model by running the script
RUN bash ./models/download-ggml-model.sh base.en

# Build the main example
RUN make

# Copy the Flask app to the container (assuming it's in the src directory)
COPY app.py /app/

# Install Flask and Gunicorn for the Python app
RUN pip install Flask gunicorn

# Expose the port the app runs on
EXPOSE 5000

# Start the Flask application with Gunicorn, adjust the worker number and timeout as needed
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "--chdir", "/app", "app:app"]
