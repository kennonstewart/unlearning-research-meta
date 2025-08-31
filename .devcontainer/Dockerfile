# Start from the same Python base image you were using
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Switch to root user to install packages
USER root

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    git \
    curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Switch back to the non-root vscode user
USER vscode