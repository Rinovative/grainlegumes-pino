# ==========================================
# 🧠 GrainLegumes_PINO
# Base: CUDA 12.1 + cuDNN8 on Ubuntu 22.04
# ==========================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ----------------------------------------------------------------------
# 🧩 System layer
# ----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl bzip2 git wget build-essential ca-certificates \
        openssh-client tini && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# 🧬 Micromamba Installation
# ----------------------------------------------------------------------
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# ----------------------------------------------------------------------
# 👤 Non-root user, permission fix
# ----------------------------------------------------------------------
RUN useradd -m -u 1000 -s /bin/bash mambauser && \
    mkdir -p \
      /home/mambauser/workspace/data \
      /home/mambauser/workspace/data_generation/data \
      /home/mambauser/workspace/model_training/data && \
    chown -R mambauser:mambauser /home/mambauser/workspace && \
    chmod -R a+rwX /home/mambauser/workspace

# ----------------------------------------------------------------------
# 📦 Environment creation
# ----------------------------------------------------------------------
COPY --chown=mambauser:mambauser environment.yml /tmp/environment.yml
USER root
RUN micromamba env create -f /tmp/environment.yml -y && \
    micromamba clean --all --yes
USER mambauser

# ----------------------------------------------------------------------
# 🧠 Default execution environment
# ----------------------------------------------------------------------
ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["micromamba", "run", "-n", "grainlegumes-pino", "/bin/bash", "-c"]

# ----------------------------------------------------------------------
# 📂 Copy source and install package
# ----------------------------------------------------------------------
COPY --chown=mambauser:mambauser . .
USER root
RUN micromamba run -n grainlegumes-pino pip install -e .
USER mambauser

# ----------------------------------------------------------------------
# ⚙️ Runtime — tini + init_permissions
# ----------------------------------------------------------------------
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >> /home/mambauser/.bashrc && \
    echo 'micromamba activate grainlegumes-pino' >> /home/mambauser/.bashrc

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]