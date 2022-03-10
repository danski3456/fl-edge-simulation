FROM python:3.8

WORKDIR /workspace
ENV PYTHONPATH="${PYTHONPATH}:/workspace/"

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt