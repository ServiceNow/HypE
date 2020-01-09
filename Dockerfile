FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
ARG PIP_EXTRA_INDEX_URL
ARG USER
ARG UID
RUN apt-get update && apt-get install -y wget curl bzip2 git
RUN pip install numpy==1.14.5
RUN adduser --uid $UID $USER
WORKDIR /main_code
COPY . .
