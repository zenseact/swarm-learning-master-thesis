env-name = conda-environment

black:
	black .

env-export:
	conda env export --from-history | grep -v "^prefix: " > $(env-name).yml>$(env-name).yml 

env-import:
	conda env create -n swarm-learning --file $(env-name).yml

env-update:
	conda env update --file $(env-name).yml --prune
