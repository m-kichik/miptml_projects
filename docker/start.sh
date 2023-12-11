#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus device=1 \
    -v /home/kichik_mg/miptml_projects/vehicles_classification/:/home/docker_miptml/vehicles_classification \
    --name miptml miptml:latest "/bin/bash"