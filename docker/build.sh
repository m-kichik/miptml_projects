#!/bin/bash

echo "Building container"
docker build . \
    -f Dockerfile \
    -t miptml:latest \
    --progress plain 

