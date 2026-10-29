# TeachLLM

Скрипты для SFT обучения Qwen-моделей на датасетах MegaPrompt и оценки поведения в MA-Craftax.

## Требования

- Linux/macOS
- Python 3.10+
- `python3-venv`
- GPU для обучения (опционально, CPU-режим поддерживается для части утилит)

Для SR/eval в среде требуется соседняя структура репозиториев:

- `.../project/TeachLLM`
- `.../project/MegaPrompt`
- `.../project/MultyAgentCrafText/MA-Craftax`

Именно такую структуру ожидают дефолтные пути в `train_sft_qwen_3.py`.

## Быстрый старт

```bash
cd /home/irazumovskii/project/TeachLLM
chmod +x scripts/install.sh scripts/run.sh
./scripts/install.sh
```

Опционально:

- `./scripts/install.sh --with-eval-deps` — ставит CPU-версию JAX для eval-сценариев.
- `./scripts/install.sh --with-comet` — ставит `comet_ml`.

## Запуск

Единая точка запуска:

```bash
./scripts/run.sh --help
```

### 1) Обучение (SFT)

```bash
./scripts/run.sh train \
  --data_dir /home/irazumovskii/project/MegaPrompt/out/expert_prompts_tree_level1_role_action_balanced_train \
  --streaming_jsonl \
  --glob "*.jsonl.gz" \
  --output_dir /home/irazumovskii/project/runs/sft-qwen-tree \
  --model Qwen/Qwen2.5-3B-Instruct \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-4 \
  --precision fp32 \
  --attn_implementation eager \
  --no-dataloader_pin_memory \
  --no-tf32
```

Чтобы увидеть все параметры:

```bash
./scripts/run.sh train --help
```

### 2) Offline accuracy на датасете

```bash
./scripts/run.sh eval-accuracy --help
```

### 3) Rollout визуализация в env

```bash
./scripts/run.sh rollout --help
```

## Полезные замечания

- Если виртуальное окружение создано в нестандартном месте, используйте:
  `VENV_DIR=/path/to/venv ./scripts/install.sh`
- `run.sh` автоматически активирует `.venv`, если она существует.
- Для совместного запуска PyTorch + JAX в проблемных конфигурациях можно использовать `JAX_PLATFORMS=cpu`.
