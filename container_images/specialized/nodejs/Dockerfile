# LLMCore Sandbox Node.js Image
# =============================
# Node.js development environment for agent sandboxes.
#
# Extends the base image with:
# - Node.js 22 LTS with npm
# - Git for version control
# - Build tools for native modules
#
# Usage:
#   docker build -t llmcore-sandbox-nodejs:1.0.0 .

ARG BASE_TAG=1.0.0
FROM llmcore-sandbox-base:${BASE_TAG}

USER root

LABEL org.llmcore.image.name="llmcore-sandbox-nodejs" \
      org.llmcore.image.version="1.0.0" \
      org.llmcore.image.tier="specialized" \
      org.llmcore.image.parent="llmcore-sandbox-base:1.0.0" \
      org.llmcore.image.description="Node.js development environment for LLMCore agent sandboxes" \
      org.llmcore.image.capabilities="nodejs,shell,git"

ENV NODE_VERSION=22 \
    NPM_CONFIG_CACHE=/home/sandbox/.npm \
    LLMCORE_IMAGE_TIER=specialized

# Install Node.js from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash - && \
    apt-get install -y --no-install-recommends \
        nodejs \
        git \
        build-essential \
        && \
    find / -xdev -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure npm as sandbox user
USER sandbox
RUN mkdir -p ~/.npm && npm config set prefix ~/.local

# Copy capabilities manifest
USER root
COPY capabilities.json /etc/llmcore/capabilities.json

USER sandbox
WORKDIR /home/sandbox/work

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD [ "node", "-e", "console.log('healthy')" ]

CMD ["node"]
