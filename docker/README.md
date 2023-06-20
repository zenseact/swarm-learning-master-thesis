Build a local docker image

```docker
docker build -t <name_of_image> -f docker/Dockerfile .
```

Save a local docker image to `.tar`

```docker
docker save -o <path for generated tar file> <image name>
```

Send the `.tar` file to the other host and load it by

```docker
docker load -i <path to image tar file>
```


```docker
docker run -v /mnt/ZOD:/mnt/ZOD <image name> --cpus=<num_cpus> -m <num_of_GB>GB
```