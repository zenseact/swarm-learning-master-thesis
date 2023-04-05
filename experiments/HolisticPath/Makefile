.PHONY: format

run:
ifdef config
	nohup python ../../platform/run_platform.py --config=$(config) &
else
	nohup python ../../platform/run_platform.py --config=config.json &
endif


find:
	pgrep -fa "python.*run_platform.py.*config*"
