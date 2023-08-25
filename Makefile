############################
#### Exporting variables ###
############################
export REPOSITORY_NAME:=$(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g')
export BRANCH_NAME=$(shell git rev-parse --abbrev-ref HEAD | sed "s/\//-/g")

################
#### Develop ###
################

.PHONY: pre
## Run pre-commit on all files
pre:
	pre-commit run --all-files

################
# Python setup #
################

.PHONY: install-src
## Install Python Package in editable mode with src dependencies
install-src:
	python -m pip install -e ".[default]"

.PHONY: install-dev
## Install Python Package in editable mode with all the dev dependencies
install-dev:
	python -m pip install -e ".[dev]"

## Install GPU related torch packages
install-gpu:
	python -m pip install cuda-python
	python -m pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu118