#!/bin/bash

set -e

PYTHON_VERSION="3.12"

if [ -n "${SLURM_TMPDIR}" ]; then
    echo "Using SLURM_TMPDIR: ${SLURM_TMPDIR}"
    ENV_DIR="${SLURM_TMPDIR}/env"
else
    ENV_DIR="./env"
fi

REQUIREMENTS_FILE="./requirements.txt"

set -x
module load "python/${PYTHON_VERSION}"
set +x

function env-start() {
    if [ ! -d "${ENV_DIR}" ]; then
        echo "Creating virtual environment in ${ENV_DIR}"
        set -x
        virtualenv --no-download "${ENV_DIR}"
        set +x
    fi
    source "${ENV_DIR}/bin/activate"
}
env-start

function pip-install() {
    set -x
    pip install --no-index --upgrade "$@"
    set +x
}
pip-install pip

if [ -n "${SLURM_JOB_ID}" ]; then
    pip-install .
else
    pip-install --editable .
fi
