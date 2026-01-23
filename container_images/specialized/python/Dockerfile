# LLMCore Sandbox Python Image
# ============================
# Python development environment for agent sandboxes.
#
# Extends the base image with:
# - Python 3.12 with pip, venv
# - Git for version control
# - Common Python development tools
#
# Usage:
#   docker build -t llmcore-sandbox-python:1.0.0 .
#
# Arguments:
#   BASE_TAG - Base image version (default: 1.0.0)

ARG BASE_TAG=1.0.0
FROM llmcore-sandbox-base:${BASE_TAG}

# Switch to root for installations
USER root

# Labels for image metadata
LABEL org.llmcore.image.name="llmcore-sandbox-python" \
      org.llmcore.image.version="1.0.0" \
      org.llmcore.image.tier="specialized" \
      org.llmcore.image.parent="llmcore-sandbox-base:1.0.0" \
      org.llmcore.image.description="Python development environment for LLMCore agent sandboxes" \
      org.llmcore.image.capabilities="python,shell,git"

# Environment configuration
ENV PYTHON_VERSION=3.12 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/home/sandbox/.venv \
    LLMCORE_IMAGE_TIER=specialized

# Update PATH for virtual environment
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install Python and development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        build-essential \
        && \
    # Security: Remove SUID/SGID bits from newly installed packages
    find / -xdev -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create virtual environment as sandbox user
USER sandbox
RUN python3 -m venv ${VIRTUAL_ENV}

# Upgrade pip and install common packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        requests \
        httpx \
        aiohttp \
        pyyaml \
        toml \
        python-dotenv \
        rich \
        && \
    # Clean pip cache
    rm -rf ~/.cache/pip

# Switch back to root to copy capabilities manifest
USER root

# Copy capabilities manifest
COPY capabilities.json /etc/llmcore/capabilities.json

# Switch to non-root user
USER sandbox

# Set working directory
WORKDIR /home/sandbox/work

# Health check - verify Python is functional
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD [ "python3", "-c", "import sys; print('healthy')" ]

# Default command
CMD ["python3"]
