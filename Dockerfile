
FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
# RUN apt-get update --fix-missing
# RUN apt-get update -y
ARG PIP_EXTRA_INDEX_URL

RUN apt-get update && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion vim emacs
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libnccl2=2.0.5-2+cuda8.0 \
         libnccl-dev=2.0.5-2+cuda8.0 \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*
     RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --no-check-certificate --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda install python=3.6.6
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN conda install tensorflow-gpu
RUN pip install --upgrade pip



# Install some python dependences I need in my project
RUN pip install numpy==1.14.5 


RUN pip install spacy
RUN python -m spacy download en
RUN pip install scipy


# add my user
RUN adduser --uid 12229 bahare
RUN pip install jupyter
RUN pip install requests
RUN conda install -c dglteam dgl-cuda10.0 
RUN pip install dgl
RUN pip install rdflib
RUN pip install pandas
WORKDIR /main_code
COPY . .
