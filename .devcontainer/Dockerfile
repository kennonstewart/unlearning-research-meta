# Start from the same Python base image you were using
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Switch to root user to install packages
USER root

# Corrected and optimized RUN command
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
    # List all packages to install here
    git \
    curl \
    git-lfs && \
    # Now run the setup scripts and cleanup in the same layer
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install && \
    # Clean up APT caches
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Switch back to the non-root vscode user
USER vscode