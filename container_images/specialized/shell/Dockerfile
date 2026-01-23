# LLMCore Sandbox Shell Image
# ===========================
# Shell scripting environment for agent sandboxes.
#
# Extends the base image with:
# - Multiple shells (bash, zsh)
# - JSON/YAML processing (jq, yq)
# - Text processing tools
# - HTTP client tools
#
# Usage:
#   docker build -t llmcore-sandbox-shell:1.0.0 .

ARG BASE_TAG=1.0.0
FROM llmcore-sandbox-base:${BASE_TAG}

USER root

LABEL org.llmcore.image.name="llmcore-sandbox-shell" \
      org.llmcore.image.version="1.0.0" \
      org.llmcore.image.tier="specialized" \
      org.llmcore.image.parent="llmcore-sandbox-base:1.0.0" \
      org.llmcore.image.description="Shell scripting environment for LLMCore agent sandboxes" \
      org.llmcore.image.capabilities="shell,jq,curl"

ENV LLMCORE_IMAGE_TIER=specialized

# Install shell tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        zsh \
        jq \
        xmlstarlet \
        bc \
        parallel \
        rsync \
        zip \
        unzip \
        tree \
        && \
    find / -xdev -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install yq (YAML processor)
RUN curl -fsSL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 \
        -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

# Copy capabilities manifest
COPY capabilities.json /etc/llmcore/capabilities.json

USER sandbox
WORKDIR /home/sandbox/work

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD [ "/bin/bash", "-c", "echo 'healthy'" ]

CMD ["/bin/bash"]
