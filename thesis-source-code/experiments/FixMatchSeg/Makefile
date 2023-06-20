.PHONY: format

FORCE ?= False
CONFIG ?= config.json
NAME ?= NULL

run:
	nohup python ../../platform/run_platform.py --config=$(CONFIG) --force=$(FORCE) --name=$(NAME)&

find:
	pgrep -fa "python.*run_platform.py.*config*"
