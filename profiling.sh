#!/bin/bash
#SBATCH --job-name=profiling
#SBATCH --output=profiling.out
#SBATCH --error=profiling.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e

source setup.sh

python examples/tabpfn_for_regression_repeated.py --fit-mode fit_preprocessors
python examples/tabpfn_for_regression_repeated.py --fit-mode fit_with_cache
python examples/tabpfn_for_regression_repeated.py --fit-mode low_memory
python examples/tabpfn_for_regression_repeated.py --fit-mode parallel
