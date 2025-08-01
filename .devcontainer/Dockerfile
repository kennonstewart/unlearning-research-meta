# Start from the same Python base image you were using
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Switch to root user to install packages
USER root

# Install a basic version of TeX Live using the system package manager
# This avoids the ghcr.io feature mechanism entirely
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive 

# Switch back to the non-root vscode user
USER vscode