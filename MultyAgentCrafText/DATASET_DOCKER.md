# Сбор MAPPO-траекторий в Docker

Для запуска нужны Docker Compose, NVIDIA GPU и NVIDIA Container Toolkit.
Каталоги `CrafText` и `MultyAgentCrafText` должны находиться рядом.

Из каталога `MultyAgentCrafText` укажите путь к чекпоинту на машине с Docker
и Hugging Face dataset-репозиторий:

```bash
cd /ПУТЬ/К/ma-craftext/MultyAgentCrafText
export COLLECT_CHECKPOINT="/ПОЛНЫЙ/ПУТЬ/mappo_model_latest.pkl"
export HF_DATASET_REPO_ID="ваш-логин/название-датасета"
```

Введите токен Hugging Face с правом записи в репозиторий (команда для bash):

```bash
read -r -s -p "HF token: " HF_TOKEN; echo
export HF_TOKEN
```

Соберите образ и запустите пробный сбор четырёх успешных траекторий:

```bash
docker compose -f compose.collect.yaml build collect
docker compose -f compose.collect.yaml run --rm collect
```

Для основного сбора задайте параметры и повторите запуск:

```bash
export COLLECT_GPU_ID=0
export COLLECT_NUM_TRAJECTORIES=100
export COLLECT_NUM_ENVS=2
export COLLECT_ROLLOUT_STEPS=128
export COLLECT_MAX_STEPS=2000
export COLLECT_SEED=0

docker compose -f compose.collect.yaml run --rm collect --max-attempts 500
```

`--max-attempts` ограничивает количество попыток; без него сбор продолжается
до получения заданного числа успешных траекторий. Можно также передать
`--greedy` или `--task-idx 0`. При нехватке памяти GPU уменьшите
`COLLECT_NUM_ENVS` или `COLLECT_ROLLOUT_STEPS`.

Сохраняются только успешные эпизоды. Их части загружаются на Hugging Face и
удаляются локально после успешной отправки. Новый dataset-репозиторий создаётся
приватным. При достижении лимита попыток частичный набор публикуется с
`complete: false`, если удалось собрать хотя бы одну траекторию.

При ошибке незагруженные файлы остаются в Docker volume. Просмотреть их можно
командой:

```bash
docker compose -f compose.collect.yaml run --rm --entrypoint bash collect \
  -c 'ls -lahR /staging'
```

Автоматического продолжения прерванного сбора нет. Для восстановления полных
наблюдений из сохранённых частей используйте `baselines/read_mappo_parts.py`.
