#
# Docker file to create a Python Environment
#

FROM ubuntu:20.04 as buildoptimizer
ARG GRB_VERSION=9.5.1
ARG GRB_SHORT_VERSION=9.5

# install gurobi package and copy the files
WORKDIR /opt

RUN apt-get update \
    && apt-get install --no-install-recommends -y\
    ca-certificates  \
    wget \
    && update-ca-certificates \
    && wget -v https://packages.gurobi.com/${GRB_SHORT_VERSION}/gurobi${GRB_VERSION}_linux64.tar.gz \
    && tar -xvf gurobi${GRB_VERSION}_linux64.tar.gz  \
    && rm -f gurobi${GRB_VERSION}_linux64.tar.gz \
    && mv -f gurobi* gurobi \
    && rm -rf gurobi/linux64/docs

# After the file renaming, a clean image is build
# FROM python:3.8 AS packageoptimizer
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 as packageoptimizer

ARG GRB_VERSION=9.5.1

LABEL vendor="Gurobi"
LABEL version=${GRB_VERSION}


ENV DEBIAN_FRONTEND=noninteractive
# update system and certificates
RUN apt-get update \
    && apt-get install --no-install-recommends -y\
    ca-certificates  \
    p7zip-full \
    zip \
    git \
    build-essential \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.7-dev python3-pip python3-setuptools python3.7-distutils \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.7 /usr/bin/python

WORKDIR /opt/gurobi
COPY --from=buildoptimizer /opt/gurobi .

ENV GUROBI_HOME /opt/gurobi/linux64
ENV PATH $PATH:$GUROBI_HOME/bin
ENV LD_LIBRARY_PATH $GUROBI_HOME/lib


WORKDIR /opt/gurobi/linux64
#run the setup
RUN python setup.py install

ENV PYTHONPATH "${PYTHONPATH}:/workspace"

COPY infrastructure/gurobi.lic /opt/gurobi/gurobi.lic

RUN python -m pip install --upgrade pip

COPY infrastructure/base.txt /tmp/pip-req/base.txt
RUN python -m pip install -r /tmp/pip-req/base.txt
COPY infrastructure/dev.txt /tmp/pip-req/dev.txt
RUN python -m pip install -r /tmp/pip-req/dev.txt

WORKDIR /workspace

