# LLMCore Sandbox Base Image
# ==========================
# Minimal, security-hardened Ubuntu base image for agent sandboxes.
#
# Security features:
# - Non-root user with limited capabilities
# - Read-only filesystem capability markers
# - Dropped Linux capabilities
# - No SUID/SGID binaries
# - AppArmor/seccomp profiles supported
#
# Usage:
#   docker build -t llmcore-sandbox-base:1.0.0 .
#
# Arguments:
#   UBUNTU_VERSION - Base Ubuntu version (default: 24.04)
#   SANDBOX_USER   - Non-root user name (default: sandbox)
#   SANDBOX_UID    - User ID (default: 1000)
#   SANDBOX_GID    - Group ID (default: 1000)

ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION}

# Build arguments
ARG SANDBOX_USER=sandbox
ARG SANDBOX_UID=1000
ARG SANDBOX_GID=1000

# Labels for image metadata
LABEL org.llmcore.image.name="llmcore-sandbox-base" \
      org.llmcore.image.version="1.0.0" \
      org.llmcore.image.tier="base" \
      org.llmcore.image.description="Security-hardened Ubuntu base image for LLMCore agent sandboxes" \
      org.llmcore.image.capabilities="shell" \
      org.opencontainers.image.source="https://github.com/llmcore/llmcore"

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC \
    PATH="/home/${SANDBOX_USER}/.local/bin:${PATH}" \
    HOME="/home/${SANDBOX_USER}" \
    LLMCORE_SANDBOX=1 \
    LLMCORE_IMAGE_TIER=base

# Install minimal packages and security updates
# Note: We intentionally keep this minimal for the base image
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        bash \
        coreutils \
        findutils \
        grep \
        gawk \
        sed \
        tar \
        gzip \
        bzip2 \
        xz-utils \
        file \
        less \
        && \
    # Security: Remove SUID/SGID bits
    find / -xdev -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    # Remove potentially dangerous commands
    rm -f /usr/bin/chfn /usr/bin/chsh /usr/bin/newgrp /usr/bin/write 2>/dev/null || true

# Create non-root user
RUN groupadd -g ${SANDBOX_GID} ${SANDBOX_USER} && \
    useradd -u ${SANDBOX_UID} -g ${SANDBOX_GID} -m -s /bin/bash ${SANDBOX_USER}

# Create standard directories
RUN mkdir -p /home/${SANDBOX_USER}/.local/bin \
             /home/${SANDBOX_USER}/.config \
             /home/${SANDBOX_USER}/work \
             /etc/llmcore && \
    chown -R ${SANDBOX_USER}:${SANDBOX_USER} /home/${SANDBOX_USER}

# Copy capabilities manifest (will be used by the selector)
COPY capabilities.json /etc/llmcore/capabilities.json

# Set working directory
WORKDIR /home/${SANDBOX_USER}/work

# Switch to non-root user
USER ${SANDBOX_USER}

# Health check - verify basic shell functionality
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=2 \
    CMD [ "/bin/bash", "-c", "echo 'healthy'" ]

# Default command
CMD ["/bin/bash"]
