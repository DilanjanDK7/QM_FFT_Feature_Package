# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Copy the requirements file into the container at /app
# We copy only pyproject.toml first to leverage Docker cache for dependencies
COPY pyproject.toml ./

# Install build dependencies and then project dependencies
# Using --no-cache-dir can reduce image size
RUN pip install --no-cache-dir setuptools wheel
RUN pip install --no-cache-dir "."
# Install optional viz dependencies if needed (uncomment if required)
# RUN pip install --no-cache-dir ".[viz]"

# Copy the rest of the application code into the container
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Make port 80 available to the world outside this container (if needed for web apps/APIs)
# EXPOSE 80

# Define environment variables (if needed)
# ENV NAME World

# Run python when the container launches (optional, can be overridden)
# CMD ["python", "your_script.py"] 