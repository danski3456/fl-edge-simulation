FROM gurobi/optimizer:latest
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

COPY infrastructure/gurobi.lic /opt/gurobi/gurobi.lic

COPY infrastructure/base.txt /tmp/pip-req/base.txt
RUN pip install -r /tmp/pip-req/base.txt
COPY infrastructure/dev.txt /tmp/pip-req/dev.txt
RUN pip install -r /tmp/pip-req/dev.txt

WORKDIR /workspace