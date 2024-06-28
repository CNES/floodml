FROM rapidsai/rapidsai:21.08-cuda11.0-base-ubuntu20.04-py3.8

RUN apt-key adv --keyserver keyserver.ubuntu.com \
        --recv-keys A4B469963BF863CC && \
    apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgeos-dev \
        libproj-dev \
        proj-data \ 
        proj-bin \
        curl && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN pip install -r \
    requirements-docker.txt --no-cache-dir \
    && rm -f requirements-docker.txt 

COPY --from=osgeo/gdal:latest /usr/local /usr/local

ENV PATH="/opt/conda/envs/rapids/bin:${PATH}"

RUN /opt/conda/envs/rapids/bin/pip install cartopy

COPY . /rapids

ENTRYPOINT ["/opt/conda/envs/rapids/bin/python", "RDF-3-inference.py"]
