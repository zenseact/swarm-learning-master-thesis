.PHONY: format

FORCE ?= False
CONFIG ?= config.json

run:
	nohup python ../../platform/run_platform.py --config=$(CONFIG) --force=$(FORCE) &

find:
	pgrep -fa "python.*run_platform.py.*config*"
