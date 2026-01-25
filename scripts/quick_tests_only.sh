#!/usr/bin/env bash

dir_llmcore="."
dir_venv="${dir_llmcore}/venv"

cd "${dir_llmcore}"

source "${dir_venv}/bin/activate"

pytest tests/ --ignore=tests/api/test_context_info_introspection.py --ignore=tests/api/test_external_rag_integration.py

deactivate
