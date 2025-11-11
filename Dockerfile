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
# 👤 Flexible non-root user (UID/GID passed at build time)
# ----------------------------------------------------------------------
# Allows you to build with:  docker build --build-arg USER_ID=$(id -u) -t grainlegumes-pino .
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} mambauser && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash mambauser

# --- 🔧 Ensure workspace and subfolders exist + correct ownership ---
RUN mkdir -p \
    /home/mambauser/workspace/data \
    /home/mambauser/workspace/data_generation/data \
    /home/mambauser/workspace/model_training/data && \
    chown -R ${USER_ID}:${GROUP_ID} /home/mambauser/workspace && \
    chmod -R u+rwX /home/mambauser/workspace

USER mambauser
WORKDIR /home/mambauser/workspace

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
# 🔹 Auto-activate env in login shell
# ----------------------------------------------------------------------
RUN echo 'eval "$(micromamba shell hook -s bash)" && micromamba activate grainlegumes-pino' >> ~/.bashrc

# ----------------------------------------------------------------------
# ⚙️ Runtime — tini + init_permissions
# ----------------------------------------------------------------------
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-l"]
