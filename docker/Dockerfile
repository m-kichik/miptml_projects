FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN python3 -m pip install scikit-learn wandb

# add user and his password
ARG USER=docker_miptml
ARG UID=1054
ARG GID=1054
# default password
ARG PW=user
RUN id

# Create the user
# RUN sudo useradd -m docker_yolov8 --uid=1054 && echo "docker_yolov8:user1" | sudo chpasswd && sudo adduser docker_yolov8
RUN useradd -ms /bin/bash docker_miptml --uid=1054
RUN id
USER docker_miptml
WORKDIR /home/docker_miptml
USER 1054:1054
