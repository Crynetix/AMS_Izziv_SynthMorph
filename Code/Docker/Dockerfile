# Use the NVIDIA CUDA base image with CUDA 11.8, cuDNN 8, and Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables to suppress prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        wget \
        git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . /app

# Set the default command to bash
CMD ["/bin/bash"]

# Set the entrypoint to your training script
ENTRYPOINT ["python3", "train.py"]
