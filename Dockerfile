# Use the nvidia/cuda image as the base image
FROM pytorch/pytorch:latest

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    bash \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    transformers \
    lightning \
    deepspeed \
    scikit-learn \
    loguru \
    pre-commit