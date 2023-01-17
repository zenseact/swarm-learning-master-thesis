# Development

We suggest that you use Anaconda and the provided environments. If you have not installed Anaconda please visit https://docs.conda.io/projects/conda/en/latest/user-guide/install/

## Getting started

To import our environment for the first time:

```bash
conda env create -n swarm-learning --file conda-environment.yml
```

## Exporting a new environment

If you are developing and adding new requirements/dependencies for the conda environment. Ensure to export this updated environment.

```makefile
make env-export
```

## Updating your environment

If there have been new dependencies added that are not in your local conda environment. You can update and prune your environment using

```makefile
make env-update
```

## Code submission

When committing to this repository, ensure that you have run black to properly format your code.

```makefile
make black
```
