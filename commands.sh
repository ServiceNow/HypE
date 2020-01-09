# Build image
docker build -t volatile-images.borgy.elementai.net/hype/hype-debug:latest --build-arg UID=$(id -u `whoami`) --build-arg USER=`whoami` .

# Publish image.
docker push volatile-images.borgy.elementai.net/hype/hype-debug:latest

# Run in workstation.
nvidia-docker run -it volatile-images.borgy.elementai.net/hype/hype-debug:latest /bin/bash

nvidia-docker run -it volatile-images.borgy.elementai.net/hype/hype-debug:latest python main.py -m HSimplE -lr 0.06

# Run in cluster.
borgy submit -i volatile-images.borgy.elementai.net/hype/hype-debug:latest --gpu 2 --gpu-mem 32 --mem 64 -I -- python main.py -m HSimplE