# Setting up docker-compose nodes on multiple hosts
To run a docker-compose deployment on multiple compute nodes on different host machines, one first needs to initialize a docker swarm on the master node:

```docker
docker swarm init
````

To connect the other worker nodes, copy the token printed out in the master node terminal. On each worker node host run:

```docker
docker swarm join --token <ref-token-from-master-node> 
```

In a multiple hosts deployment, ensure the correct ip addresses are specified:
- In `client_main.py`, change the `server_ip` and `agx_ip` to the corresponding IP addresses of the server node and agx node
- In `server_main.py`, set the server ip to `server_ip="0.0.0.0"`
- In `scheduler_main.py`, set the agx ip to `agx_ip="0.0.0.0"`
- In `docker-compose.yml`, set the constraints for the server node, agx node and client node to each node's corresponding ip address

Finally, also ensure that the number of client replicas in `docker-compose.yml` correspond to `NUM_CLIENTS` in `static_params.py`.

In a multiple hosts deployment, a pre-built image needs to be accessible on each worker node. Hence, start with building the right image on each worker by

```docker
docker build -f src/Dockerfile . -t <tag-of-application>
```

If bulding the docker images on the AGXes is difficult, you can build them elsewhere and then export them as .tar files:
```docker
docker save -o docker/images/scheduler.tar scheduler-app
```

Then transfer the image to the AGX node:
```docker
scp docker/images/scheduler.tar nvidia@agx4.nodes.lab.ai.se:/home/nvidia/Fleet/fleet-learning/docker/images
```

Finally, on the agx node, build image from that tar file:
```docker
docker load < docker/images/scheduler.tar
```


If you are on the AGXs and don't have permission to access the docker command, you can run

```docker
sudo chmod 666 /var/run/docker.sock
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

To undo the complete docker swarm init process, run the below command

```docker
docker swarm leave --force
``````