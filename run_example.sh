#!/usr/bin/env bash


dir_base="${PWD}"
dir_examples="${dir_base}/examples"
dir_venv="${dir_base}/venv"
dir_logs="/av/logs/llmcore"



tag_suffix="$(date +'%Y%m%d-%H%M%S')"
source ${dir_venv}/bin/activate



choice="$(find ${dir_examples} -type f -name "*.py" -printf "%f\n" | fzf -e)"
choice_extless="$(echo "${choice}" | sed -E 's/\.py//')"


f_log="${dir_logs}/llmcore_run_${choice_extless}_${tag_suffix}.log.txt"
pyfile="${dir_examples}/${choice}"

if [[ -f ${pyfile} ]]; then
    python "${pyfile}" 2>&1 | tee -a ${f_log}
fi 

deactivate
