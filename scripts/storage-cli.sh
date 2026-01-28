#!/usr/bin/env bash

dir_venv="${1:-./venv}"

if [[ -d ${dir_venv} ]]; then
    if [[ -f "${dir_venv}/bin/activate" ]]; then
        shift
        source "${dir_venv}/bin/activate"
    fi
fi

if [[ -n "${LLMCORE_CONFIG_PATH}" && -f "${LLMCORE_CONFIG_PATH}" ]]; then
    python -m llmcore.storage.cli --config "${LLMCORE_CONFIG_PATH}" $@
else
    echo "Please ensure 'LLMCORE_CONFIG_PATH' variable exists"
    echo "and points to a valid llmcore configuration file."
    echo "You can do so by 'export LLMCORE_CONFIG_PATH=PATH_TO_YOUR_CONFIG'"
fi

deactivate

