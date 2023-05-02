.PHONY: format

run:
ifdef config
	nohup python ../../platform/run_platform.py --config=$(config) --force=False &
else
	nohup python ../../platform/run_platform.py --config=config.json --force=False &
endif


find:
	pgrep -fa "python.*run_platform.py.*config*"
