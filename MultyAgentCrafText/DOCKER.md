# MAPPO training in Docker

The build context must be the parent workspace because training uses both local
projects:

- `MultyAgentCrafText/MA-Craftax`
- `CrafText`

## Prerequisites

- Docker with Compose
- NVIDIA driver and NVIDIA Container Toolkit
- a Comet API key for workspace `igorrazumvskii`
- a Hugging Face token with write access to `igor69raz/ma-craftax-mappo`

The Compose service exposes only host GPU `1` (`Tesla V100-SXM2-32GB`) to the
training container.

## Run

From the `MultyAgentCrafText` directory:

```bash
cp .env.example .env
# Add COMET_API_KEY and HF_TOKEN to .env.
docker compose build
docker compose run --rm mappo
```

The default command runs:

```bash
bash baselines/run_mappo_train.sh mappo_rnn.yaml
```

Metrics are sent to the Comet project `igorrazumvskii/ma-craftax`.
Each checkpoint is uploaded to the Hugging Face model repository
`igor69raz/ma-craftax-mappo` as `checkpoints/mappo_model_latest.pkl`. After each
upload, the repository history is squashed so previous LFS revisions do not
accumulate. The local checkpoint is then deleted from the container. Training
logs and checkpoints are not mounted to the host.

To use another config already present under `MA-Craftax/baselines/config`:

```bash
docker compose run --rm mappo another_config.yaml
```

To open a shell without starting training:

```bash
docker compose run --rm --entrypoint bash mappo
```

To verify GPU visibility:

```bash
docker compose run --rm --entrypoint python mappo \
  -c "import jax; print(jax.devices())"
```

External logging and checkpoint upload are configured through `.env`:

```dotenv
COMET_API_KEY=...
HF_TOKEN=hf_...
HF_HUB_UPLOAD=1
HF_HUB_REPO_ID=igor69raz/ma-craftax-mappo
```
