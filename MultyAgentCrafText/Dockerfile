FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY MultyAgentCrafText/docker/requirements-mappo.txt /tmp/requirements-mappo.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements-mappo.txt

COPY CrafText /workspace/CrafText
COPY MultyAgentCrafText/MA-Craftax /workspace/MultyAgentCrafText/MA-Craftax

# CrafText is imported from its source tree by craftext_adapter.py. Building its
# wheel would omit scenario YAML files required at runtime.
RUN chmod +x /workspace/MultyAgentCrafText/MA-Craftax/baselines/run_mappo_train.sh

WORKDIR /workspace/MultyAgentCrafText/MA-Craftax

RUN JAX_PLATFORMS=cpu python -c "import jax, torch, transformers, jaxmarl, comet_ml; import baselines.my_mappo; print('MAPPO imports OK')"

ENTRYPOINT ["bash", "baselines/run_mappo_train.sh"]
CMD ["mappo_rnn.yaml"]
