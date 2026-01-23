# LLMCore Sandbox Websearch Image
# ================================
# Web search and scraping environment for agent sandboxes.
#
# Extends the Python image with:
# - Headless browser (Playwright)
# - Web scraping libraries
# - Network access enabled
#
# Usage:
#   docker build -t llmcore-sandbox-websearch:1.0.0 .

ARG PARENT_TAG=1.0.0
FROM llmcore-sandbox-python:${PARENT_TAG}

USER root

LABEL org.llmcore.image.name="llmcore-sandbox-websearch" \
      org.llmcore.image.version="1.0.0" \
      org.llmcore.image.tier="task" \
      org.llmcore.image.parent="llmcore-sandbox-python:1.0.0" \
      org.llmcore.image.description="Web search and scraping environment for LLMCore agent sandboxes" \
      org.llmcore.image.capabilities="python,websearch,browser,network"

ENV LLMCORE_IMAGE_TIER=task \
    PLAYWRIGHT_BROWSERS_PATH=/home/sandbox/.cache/ms-playwright

# Install browser dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Chromium dependencies
        libnss3 \
        libnspr4 \
        libdbus-1-3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libgbm1 \
        libasound2t64 \
        libpango-1.0-0 \
        libcairo2 \
        # Firefox dependencies
        libgtk-3-0 \
        && \
    find / -xdev -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python websearch packages
USER sandbox
RUN pip install \
        playwright \
        beautifulsoup4 \
        lxml \
        selenium \
        requests-html \
        trafilatura \
        newspaper3k \
        && \
    rm -rf ~/.cache/pip

# Install Playwright browsers (Chromium only to save space)
RUN playwright install chromium

# Copy capabilities manifest
USER root
COPY capabilities.json /etc/llmcore/capabilities.json

USER sandbox
WORKDIR /home/sandbox/work

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=2 \
    CMD [ "python3", "-c", "from playwright.sync_api import sync_playwright; print('healthy')" ]

CMD ["python3"]
