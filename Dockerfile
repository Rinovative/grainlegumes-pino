# ==========================================
# GrainLegumes PINO Airflow
# CUDA 12.1 + cuDNN8 + Micromamba
# ==========================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV ENV_NAME=grainlegumes-pino

# ----------------------------------------------------------------------
# System layer
# ----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        bzip2 \
        ca-certificates \
        curl \
        git \
        openssh-client \
        tini \
        wget \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# Micromamba
# ----------------------------------------------------------------------
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# ----------------------------------------------------------------------
# Create environment
# ----------------------------------------------------------------------
COPY environment.yml /tmp/environment.yml

RUN micromamba env create -f /tmp/environment.yml -y && \
    micromamba clean --all --yes

# ----------------------------------------------------------------------
# Workspace layout
# ----------------------------------------------------------------------
RUN mkdir -p /workspace/repo /workspace/storage

WORKDIR /workspace/repo

# ----------------------------------------------------------------------
# Install project into environment
# ----------------------------------------------------------------------
COPY . /workspace/repo

RUN micromamba run -n ${ENV_NAME} pip install -e /workspace/repo

# ----------------------------------------------------------------------
# Runtime defaults
# ----------------------------------------------------------------------
ENV PATH=/opt/micromamba/envs/${ENV_NAME}/bin:$PATH
ENV PROJECT_ROOT=/workspace/repo
ENV STORAGE_ROOT=/workspace/storage
ENV DATA_ROOT=/workspace/storage/data
ENV DATASET_ROOT=/workspace/storage/data_training/raw
ENV GENERATED_DATA_ROOT=/workspace/storage/data_generation
ENV OUTPUT_ROOT=/workspace/storage/data_training/processed

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]