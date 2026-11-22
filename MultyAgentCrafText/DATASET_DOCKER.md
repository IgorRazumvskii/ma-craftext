# Локальный сбор успешных MAPPO-траекторий

Для запуска нужны Docker Compose, NVIDIA GPU и NVIDIA Container Toolkit.
Каталоги `CrafText` и `MultyAgentCrafText` должны находиться рядом.
Чекпоинт и каталог результатов должны существовать на сервере до запуска.

```bash
cd /ПУТЬ/К/ma-craftext/MultyAgentCrafText
export COLLECT_CHECKPOINT=/ПОЛНЫЙ/ПУТЬ/mappo_model_latest.pkl
export COLLECT_OUTPUT_DIR=/ПОЛНЫЙ/ПУТЬ/К/ДАТАСЕТУ
mkdir -p "$COLLECT_OUTPUT_DIR"
export COLLECT_GPU_ID=0
export COLLECT_NUM_TRAJECTORIES=100000
export COLLECT_NUM_ENVS=2
export COLLECT_MAX_STEPS=100
export COLLECT_ROLLOUT_STEPS=128

docker compose -f compose.collect.yaml build collect
docker compose -f compose.collect.yaml run -d --rm collect
```

Команда печатает ID контейнера. Логи: `docker logs -f <ID>`; Ctrl+C
прекращает просмотр, но не работу фонового контейнера. Для пробного запуска
перед командой задайте `COLLECT_NUM_TRAJECTORIES=4` и при необходимости
передайте `--max-attempts 100`. Без лимита попыток сбор продолжается до
получения заданного числа успешных траекторий. При нехватке памяти уменьшите
`COLLECT_NUM_ENVS` или `COLLECT_ROLLOUT_STEPS`.

Лимит — 100 совместных шагов среды: для трёх агентов это до 100 действий на
агента или 300 действий суммарно. `COLLECT_MAX_STEPS` может уменьшить лимит,
но не увеличить его выше 100. Эпизод завершается раньше
при выполнении задачи или прекращении среды; учитываются только успешные.
`COLLECT_ROLLOUT_STEPS` задаёт длину блока вычислений, а не лимит эпизода.

Результаты хранятся на сервере в новом каталоге `collect_*` внутри
`COLLECT_OUTPUT_DIR`: Parquet с переходами в `*_parts`, отдельные Parquet
`*_terminal_states` с полным состоянием мира (`state/...`) и наблюдениями
агентов (`terminal_obs`) после успешного действия и до автоматического сброса,
а также embeddings и итоговый `*.meta.json`. Состояние соединяется с переходами
по `trajectory_id`. Формат переходов совместим с TeachLLM. Функции
`iter_parts` и `iter_terminal_states` в `baselines/read_mappo_parts.py`
читают оба набора. Данные не загружаются на Hugging Face; загрузка исходной
модели кодирования текста для работы среды при первом запуске может требовать
сети и кэшируется в Docker volume.

Метаданные создаются после завершения сбора. При прерывании закрытые Parquet
шарды остаются в каталоге, но открытый шард и незавершённые эпизоды могут быть
потеряны; автоматического продолжения сбора нет. Для 100 000 эпизодов требуется
заранее проверить свободное место: полные состояния мира существенно
увеличивают размер данных.
