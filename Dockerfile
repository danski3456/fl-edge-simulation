FROM python:3.8

RUN apt-get update && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG "en_US.UTF-8"
ENV LANGUAGE "en_US.UTF-8"
ENV LC_ALL "en_US.UTF-8"

WORKDIR /workspace
ENV PYTHONPATH="${PYTHONPATH}:/workspace/"

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt