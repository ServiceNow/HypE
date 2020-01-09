# Build image
docker build -t volatile-images.borgy.elementai.net/hype/hype-debug:latest --build-arg UID=$(id -u `whoami`) --build-arg USER=`whoami` .

# Publish image.
docker push volatile-images.borgy.elementai.net/hype/hype-debug:latest

# Run in workstation.
nvidia-docker run -it --name shype1 shype1 /bin/bash

# Run in cluster.
borgy submit -i volatile-images.borgy.elementai.net/hype/hype-debug:latest --gpu 2 --gpu-mem 32 --mem 64 -I -- python main.py -m HSimplE
