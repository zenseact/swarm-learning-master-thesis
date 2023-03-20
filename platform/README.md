# Decentralised Learing & ZOD Benchmark platform
This ML platform is a capable and customisable tool for building and training machine learning models. It takes a configuration object with various parameters, such as data location, image size, model type, and loss function, to simplify the process of building and training models. The platform can be used for central and decentralized training and allows access to data without running any training. The platform makes it easy to build and train machine learning models with minimal coding and configuration. The platform enables quick iteration, testing and benchmarking of tasks on ZOD in different learning environments.


# Usage & Development
The usage of the platform is very simple, although at this stage also very limited. There are introduction notebook that describe how the platform works and how to develop using it.

- [Example 0: The basics of the ML platform](/platform/example-0-platform.ipynb)
- [Example 1: Working with data](/platform/example-1-data.ipynb)

## Logging 
The platform logs system debug information as well as training progress in real time. Logging consists of log files `.log` and tensorboard logs that can be read during training using tensorboard.

### Tensorboard
The platform uses Tensorboard for logging and logs various progress and metrics to ensure a smooth run. Data is stored in `runs` and a `run` is a directory created that represents an instance of the platform and its results. Runs are named in the following format `%Y-%m-%d_%H:%M:%S`. The following is logged to tensorboard:

**Scalars**
- Centralised Batch Loss [Training]
- Centralised Epoch Loss [Training & Validation]
- Federated Client Batch Loss [Training]
- Federated Client Epoch Loss [Training & Validation]
- Federated Global Model Global Round Loss [Training & Validation]
- Swarm Client Batch Loss [Training]
- Swarm Client Epoch Loss [Training & Validation]

**Other**
- Swarm network topology figure per "global round"
- Swarm & Federated client data frequency distribution charts
- Platform configuration file for reproducibility

### Logfiles
In the run folders we also store logfiles. Generally these log files are set to log at the `DEBUG` level and will include detailed information about the process. Generally most information is stored in `platform.log`, however, in async functions that use ray backend, such as federated and swarm clients, separate log files will contain more detailed information `swarm.log` and `federated.log`. 

For decentralised learning, clients will also have their own logs in the format `swarm_{client_id}.log`. These logs will contain similar logging as in the centralised case.

## Python Environment
We suggest that you use Anaconda and the provided environments. If you have not installed Anaconda please visit https://docs.conda.io/projects/conda/en/latest/user-guide/install/

I have not provided any conda env file yet. So I urge anyone using this to try to replicate or build their own.


# Code submission

When committing to this repository, ensure that you have run black to properly format your code.

```makefile
make black
```

