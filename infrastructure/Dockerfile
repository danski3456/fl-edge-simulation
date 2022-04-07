ARG IMAGEFILE=nvidia/cuda:11.4.2-devel-ubuntu20.04
FROM ${IMAGEFILE} 

ARG UID
ENV PYTHONDONTWRITEBYTECODE 1 \  # Keeps Python from generating .pyc files in the container
ENV PYTHONUNBUFFERED 1 # Turns off buffering for easier container logging
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common 
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y build-essential software-properties-common \
    git python3-dev python3-pip python3-setuptools python3-distutils python3-pygit2 libtinfo5 libgit2-dev less locales ffmpeg libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG "en_US.UTF-8"
ENV LANGUAGE "en_US.UTF-8"
ENV LC_ALL "en_US.UTF-8"
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY infrastructure/base.txt /tmp/pip-req/base.txt
RUN pip install -r /tmp/pip-req/base.txt
COPY infrastructure/dev.txt /tmp/pip-req/dev.txt
RUN pip install -r /tmp/pip-req/dev.txt

ENV PYTHONPATH "${PYTHONPATH}:/workspace"
WORKDIR /workspace