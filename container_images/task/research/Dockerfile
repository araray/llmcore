# LLMCore Sandbox Research Image
# ==============================
# Research environment for agent sandboxes.
#
# Extends the Python image with:
# - Document processing (pandoc, PDF tools)
# - Data analysis libraries
# - Network access for API calls
#
# Usage:
#   docker build -t llmcore-sandbox-research:1.0.0 .

ARG PARENT_TAG=1.0.0
FROM llmcore-sandbox-python:${PARENT_TAG}

USER root

LABEL org.llmcore.image.name="llmcore-sandbox-research" \
      org.llmcore.image.version="1.0.0" \
      org.llmcore.image.tier="task" \
      org.llmcore.image.parent="llmcore-sandbox-python:1.0.0" \
      org.llmcore.image.description="Research environment for LLMCore agent sandboxes" \
      org.llmcore.image.capabilities="python,research,network,pdf_tools,pandoc"

ENV LLMCORE_IMAGE_TIER=task

# Install document processing tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pandoc \
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        poppler-utils \
        ghostscript \
        wkhtmltopdf \
        && \
    find / -xdev -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python research packages as sandbox user
USER sandbox
RUN pip install \
        pandas \
        numpy \
        scipy \
        matplotlib \
        seaborn \
        beautifulsoup4 \
        lxml \
        PyPDF2 \
        python-docx \
        openpyxl \
        markdown \
        && \
    rm -rf ~/.cache/pip

# Copy capabilities manifest
USER root
COPY capabilities.json /etc/llmcore/capabilities.json

USER sandbox
WORKDIR /home/sandbox/work

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD [ "python3", "-c", "import pandas; print('healthy')" ]

CMD ["python3"]
