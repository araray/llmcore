# LLMCore Sandbox Container Images

Container images for the LLMCore agent sandbox system. These images provide secure, isolated environments for AI agents to execute code and perform tasks.

## Image Tiers

Images are organized into three tiers:

### Base Tier
Minimal, security-hardened foundation images.

| Image | Description |
|-------|-------------|
| `llmcore-sandbox-base:1.0.0` | Minimal Ubuntu 24.04 with shell tools |

### Specialized Tier
Language-specific development environments.

| Image | Description | Capabilities |
|-------|-------------|--------------|
| `llmcore-sandbox-python:1.0.0` | Python 3.12 development | python, git, pip |
| `llmcore-sandbox-nodejs:1.0.0` | Node.js 22 development | nodejs, npm, git |
| `llmcore-sandbox-shell:1.0.0` | Shell scripting | bash, zsh, jq, yq |

### Task Tier
Purpose-built images for specific workflows.

| Image | Description | Access Mode |
|-------|-------------|-------------|
| `llmcore-sandbox-research:1.0.0` | Research & document analysis | FULL |
| `llmcore-sandbox-websearch:1.0.0` | Web scraping & search | FULL |

## Quick Start

### Build All Images

```bash
cd container_images
make all
```

### Build Specific Images

```bash
# Build only base
make base

# Build Python (includes base dependency)
make python

# Build Research (includes python → base)
make research
```

### Test Images

```bash
make test
```

## Image Hierarchy

```
llmcore-sandbox-base:1.0.0
├── llmcore-sandbox-python:1.0.0
│   ├── llmcore-sandbox-research:1.0.0
│   └── llmcore-sandbox-websearch:1.0.0
├── llmcore-sandbox-nodejs:1.0.0
└── llmcore-sandbox-shell:1.0.0
```

## Capabilities

Each image includes a capabilities manifest at `/etc/llmcore/capabilities.json`:

```json
{
  "name": "llmcore-sandbox-python",
  "version": "1.0.0",
  "tier": "specialized",
  "capabilities": ["python", "shell", "git"],
  "tools": ["python3", "pip", "git"],
  "default_access_mode": "restricted"
}
```

## Access Modes

| Mode | Network | Filesystem | Use Case |
|------|---------|------------|----------|
| RESTRICTED | Blocked | Limited | Code execution, analysis |
| FULL | Enabled | Extended | Research, web access |

## Security Features

All images include:
- **Non-root user**: Runs as `sandbox` (UID 1000)
- **No SUID/SGID**: Privilege escalation binaries removed
- **Minimal packages**: Only essential tools installed
- **Health checks**: Built-in container health monitoring
- **AppArmor/seccomp ready**: Compatible with security profiles

See [SECURITY.md](SECURITY.md) for detailed security information.

## Usage with LLMCore

```python
from llmcore.agents.sandbox.images import ImageSelector, ImageRegistry

# Create registry with builtin manifests
registry = ImageRegistry()

# Select image for task
selector = ImageSelector(registry)
result = selector.select_for_task("research")
print(f"Using image: {result.image}")  # llmcore-sandbox-research:1.0.0

# Select by runtime
result = selector.select_for_runtime("python")
print(f"Using image: {result.image}")  # llmcore-sandbox-python:1.0.0
```

## Custom Images

You can extend these images for custom needs:

```dockerfile
FROM llmcore-sandbox-python:1.0.0

USER root
RUN pip install your-custom-package

# Update capabilities manifest
COPY capabilities.json /etc/llmcore/capabilities.json

USER sandbox
```

Then register with the selector:

```python
selector.add_task_mapping("custom_task", "my-custom-image:1.0.0")
```

## Resource Limits

Default resource limits by tier:

| Tier | Memory | CPU | Timeout | Processes |
|------|--------|-----|---------|-----------|
| Base | 512 MB | 1 | 5 min | 50 |
| Specialized | 1 GB | 2 | 10 min | 100 |
| Task | 2-4 GB | 2-4 | 15-30 min | 150-200 |

## Pushing to Registry

```bash
make push REGISTRY=your-registry.com
```

## License

See the main LLMCore repository for license information.
