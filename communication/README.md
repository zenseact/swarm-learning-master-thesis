# Setting up docker-compose nodes on multiple hosts
To run a docker-compose deployment on multiple compute nodes on different host machines, one first needs to initialize a docker swarm on the master node:

```docker
docker swarm init
````

To connect the other worker nodes, copy the token printed out in the master node terminal. On each worker node host run:

```docker
docker swarm join --token <ref-token-from-master-node> 

In a multiple hosts deployment, a pre-built image needs to be accessible on each worker node. Hence, start with building the right image on each worker by

```docker
docker build -f <path>/Dockerfile . -t <tag-of-application>
```

The pre-built images need to be referenced in the master node docker-compose file, together with the location of each host:

`docker-compose.yml`:
```yaml
worker-app:
  server:
  image: <local-server-img> 
  environment:
  - "contraint:node==<ip-adress-of-host>"
```

To run the full deployment, run following command on the master node:

```docker
 docker stack deploy -c docker-compose.yml <name-of-deployment>
```

To remove the deployment, run:

```docker
docker stack rm <name-of-deployment> 
```
