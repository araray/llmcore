#!/usr/bin/env bash

dir_llmcore="."
dir_venv="${dir_llmcore}/venv"

cd "${dir_llmcore}"

source "${dir_venv}/bin/activate"

export TEST_POSTGRES_URL="postgresql://tester:tester@localhost:5432/postgres"
export LLMCORE_TEST_PG_HOST=127.0.0.1
export LLMCORE_TEST_PG_PORT=5432
export LLMCORE_TEST_PG_USER=tester
export LLMCORE_TEST_PG_PASSWORD=tester
export LLMCORE_TEST_PG_DATABASE=postgres

unset LLMCORE_SKIP_PG_TESTS

pytest tests/ \
    --ignore=tests/api/test_context_info_introspection.py \
    --ignore=tests/api/test_external_rag_integration.py

deactivate
