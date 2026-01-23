# LLMCore Container Images Build System
# ======================================
#
# Usage:
#   make all          - Build all images
#   make base         - Build base image only
#   make specialized  - Build all specialized images
#   make task         - Build all task images
#   make clean        - Remove all llmcore-sandbox images
#   make test         - Test all images
#   make push         - Push all images to registry
#
# Variables:
#   REGISTRY         - Container registry (default: empty, local only)
#   VERSION          - Image version (default: 1.0.0)
#   DOCKER           - Docker command (default: docker)

# Configuration
REGISTRY ?=
VERSION ?= 1.0.0
DOCKER ?= docker
IMAGE_PREFIX := llmcore-sandbox

# Image names
BASE_IMAGE := $(IMAGE_PREFIX)-base:$(VERSION)
PYTHON_IMAGE := $(IMAGE_PREFIX)-python:$(VERSION)
NODEJS_IMAGE := $(IMAGE_PREFIX)-nodejs:$(VERSION)
SHELL_IMAGE := $(IMAGE_PREFIX)-shell:$(VERSION)
RESEARCH_IMAGE := $(IMAGE_PREFIX)-research:$(VERSION)
WEBSEARCH_IMAGE := $(IMAGE_PREFIX)-websearch:$(VERSION)

# Directory structure
BASE_DIR := base
SPECIALIZED_DIR := specialized
TASK_DIR := task

# Build order (respects dependencies)
.PHONY: all base specialized task clean test push

all: base specialized task
	@echo "All images built successfully"

# Base image (no dependencies)
base:
	@echo "Building base image: $(BASE_IMAGE)"
	$(DOCKER) build -t $(BASE_IMAGE) $(BASE_DIR)/
	@echo "Base image built: $(BASE_IMAGE)"

# Specialized images (depend on base)
specialized: base python nodejs shell
	@echo "All specialized images built"

python: base
	@echo "Building Python image: $(PYTHON_IMAGE)"
	$(DOCKER) build -t $(PYTHON_IMAGE) $(SPECIALIZED_DIR)/python/
	@echo "Python image built: $(PYTHON_IMAGE)"

nodejs: base
	@echo "Building Node.js image: $(NODEJS_IMAGE)"
	$(DOCKER) build -t $(NODEJS_IMAGE) $(SPECIALIZED_DIR)/nodejs/
	@echo "Node.js image built: $(NODEJS_IMAGE)"

shell: base
	@echo "Building Shell image: $(SHELL_IMAGE)"
	$(DOCKER) build -t $(SHELL_IMAGE) $(SPECIALIZED_DIR)/shell/
	@echo "Shell image built: $(SHELL_IMAGE)"

# Task images (depend on specialized)
task: specialized research websearch
	@echo "All task images built"

research: python
	@echo "Building Research image: $(RESEARCH_IMAGE)"
	$(DOCKER) build -t $(RESEARCH_IMAGE) $(TASK_DIR)/research/
	@echo "Research image built: $(RESEARCH_IMAGE)"

websearch: python
	@echo "Building Websearch image: $(WEBSEARCH_IMAGE)"
	$(DOCKER) build -t $(WEBSEARCH_IMAGE) $(TASK_DIR)/websearch/
	@echo "Websearch image built: $(WEBSEARCH_IMAGE)"

# Testing
test: all
	@echo "Testing images..."
	@$(DOCKER) run --rm $(BASE_IMAGE) /bin/bash -c "echo 'Base: OK'"
	@$(DOCKER) run --rm $(PYTHON_IMAGE) python3 -c "import sys; print(f'Python {sys.version}: OK')"
	@$(DOCKER) run --rm $(NODEJS_IMAGE) node -e "console.log('Node.js ' + process.version + ': OK')"
	@$(DOCKER) run --rm $(SHELL_IMAGE) /bin/bash -c "jq --version && echo 'Shell: OK'"
	@$(DOCKER) run --rm $(RESEARCH_IMAGE) python3 -c "import pandas; print('Research: OK')"
	@$(DOCKER) run --rm $(WEBSEARCH_IMAGE) python3 -c "from playwright.sync_api import sync_playwright; print('Websearch: OK')"
	@echo "All tests passed!"

# Clean up
clean:
	@echo "Removing llmcore-sandbox images..."
	-$(DOCKER) rmi $(WEBSEARCH_IMAGE) 2>/dev/null || true
	-$(DOCKER) rmi $(RESEARCH_IMAGE) 2>/dev/null || true
	-$(DOCKER) rmi $(SHELL_IMAGE) 2>/dev/null || true
	-$(DOCKER) rmi $(NODEJS_IMAGE) 2>/dev/null || true
	-$(DOCKER) rmi $(PYTHON_IMAGE) 2>/dev/null || true
	-$(DOCKER) rmi $(BASE_IMAGE) 2>/dev/null || true
	@echo "Clean complete"

# Push to registry
push: all
ifndef REGISTRY
	$(error REGISTRY is not set. Use: make push REGISTRY=your-registry.com)
endif
	@echo "Pushing images to $(REGISTRY)..."
	$(DOCKER) tag $(BASE_IMAGE) $(REGISTRY)/$(BASE_IMAGE)
	$(DOCKER) push $(REGISTRY)/$(BASE_IMAGE)
	$(DOCKER) tag $(PYTHON_IMAGE) $(REGISTRY)/$(PYTHON_IMAGE)
	$(DOCKER) push $(REGISTRY)/$(PYTHON_IMAGE)
	$(DOCKER) tag $(NODEJS_IMAGE) $(REGISTRY)/$(NODEJS_IMAGE)
	$(DOCKER) push $(REGISTRY)/$(NODEJS_IMAGE)
	$(DOCKER) tag $(SHELL_IMAGE) $(REGISTRY)/$(SHELL_IMAGE)
	$(DOCKER) push $(REGISTRY)/$(SHELL_IMAGE)
	$(DOCKER) tag $(RESEARCH_IMAGE) $(REGISTRY)/$(RESEARCH_IMAGE)
	$(DOCKER) push $(REGISTRY)/$(RESEARCH_IMAGE)
	$(DOCKER) tag $(WEBSEARCH_IMAGE) $(REGISTRY)/$(WEBSEARCH_IMAGE)
	$(DOCKER) push $(REGISTRY)/$(WEBSEARCH_IMAGE)
	@echo "Push complete"

# List images
list:
	@echo "LLMCore Sandbox Images:"
	@$(DOCKER) images | grep $(IMAGE_PREFIX) || echo "No images found"

# Image info
info:
	@echo "Base image:      $(BASE_IMAGE)"
	@echo "Python image:    $(PYTHON_IMAGE)"
	@echo "Node.js image:   $(NODEJS_IMAGE)"
	@echo "Shell image:     $(SHELL_IMAGE)"
	@echo "Research image:  $(RESEARCH_IMAGE)"
	@echo "Websearch image: $(WEBSEARCH_IMAGE)"
