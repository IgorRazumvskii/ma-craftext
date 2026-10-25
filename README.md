# ma-craftext

Multi-agent instruction following on [Craftax](https://github.com/MichaelTMatthews/Craftax). Cooperative agents are trained with MAPPO in Craftax-Coop, and goals come from [CrafText](https://github.com/AIRI-Institute/CrafText) natural-language scenarios.

Мультиагентное следование инструкциям в Craftax. Кооперативные агенты обучаются MAPPO в Craftax-Coop, цели задаются текстовыми сценариями CrafText.

## Layout / Состав

| Path | What it is |
| --- | --- |
| `CrafText/` | Instruction scenarios, checkers, and the language wrapper. |
| `MultyAgentCrafText/MA-Craftax/` | Multi-agent Craftax, the CrafText adapter, and MAPPO training. |
| `MultyAgentCrafText/` | Docker image and Compose service for training. |
| `checkpoint/` | Latest weights on the host: `mappo_model_latest.pkl`. |
| `old_MA-Craftax/` | Earlier MA-Craftax snapshot. Training does not use it. |

Keep this layout. The adapter imports `CrafText` as a sibling of `MultyAgentCrafText`.

Эту раскладку нужно сохранить: адаптер импортирует `CrafText` как соседний каталог `MultyAgentCrafText`.

## Quick start / Быстрый старт

Requirements: Docker, Docker Compose, an NVIDIA driver, and the NVIDIA Container Toolkit.

Нужны Docker, Docker Compose, драйвер NVIDIA и NVIDIA Container Toolkit.

```bash
cd MultyAgentCrafText
cp .env.example .env
```

Put a Comet API key and a Hugging Face token into `.env`. Do not commit that file.

В `.env` запишите ключ Comet и токен Hugging Face. Этот файл не коммитьте.

```bash
docker compose build
docker compose run --rm mappo
```

This starts `baselines/run_mappo_train.sh mappo_rnn.yaml`: MAPPO on `Craftax-Coop-Symbolic` with the CrafText scenario `easy_train`.

Так запускается `baselines/run_mappo_train.sh mappo_rnn.yaml`: MAPPO в `Craftax-Coop-Symbolic` со сценарием CrafText `easy_train`.

Another config from `MA-Craftax/baselines/config`:

Другой конфиг из `MA-Craftax/baselines/config`:

```bash
docker compose run --rm mappo another_config.yaml
```

Check that JAX sees the GPU:

Проверка, что JAX видит GPU:

```bash
docker compose run --rm --entrypoint python mappo \
  -c "import jax; print(jax.devices())"
```

Shell without training:

Оболочка без обучения:

```bash
docker compose run --rm --entrypoint bash mappo
```

On this machine Compose reserves host GPU `1`, and checkpoints are written to `/home/razumovsky_id/work/checkpoint`. After cloning elsewhere, change `device_ids` and the checkpoint volume in `MultyAgentCrafText/compose.yaml`.

На этой машине Compose занимает GPU `1` хоста, а чекпоинты пишет в `/home/razumovsky_id/work/checkpoint`. На другой машине поменяйте `device_ids` и volume чекпоинта в `MultyAgentCrafText/compose.yaml`.

## Where results go / Куда пишутся результаты

- Comet project `igorrazumvskii/ma-craftax`.
- Hugging Face model `igor69raz/ma-craftax-mappo`, file `checkpoints/mappo_model_latest.pkl`. Each upload squashes the repo history so old LFS revisions do not pile up.
- The same file on the host: `checkpoint/mappo_model_latest.pkl`.

Turn uploads off with `HF_HUB_UPLOAD=0` in `.env`.

Отключить выгрузку: `HF_HUB_UPLOAD=0` в `.env`.

## Train without Docker / Запуск без Docker

Python 3.10. From the repo root, install `MultyAgentCrafText/docker/requirements-mappo.txt`, then:

Python 3.10. Из корня репозитория поставьте зависимости из `MultyAgentCrafText/docker/requirements-mappo.txt`, затем:

```bash
cd MultyAgentCrafText/MA-Craftax
bash baselines/run_mappo_train.sh mappo_rnn.yaml
```

`CrafText` must stay next to `MultyAgentCrafText`. Export `COMET_API_KEY` and, if you want Hub uploads, `HF_TOKEN`.

`CrafText` должен лежать рядом с `MultyAgentCrafText`. Экспортируйте `COMET_API_KEY` и, если нужна выгрузка на Hub, `HF_TOKEN`.
