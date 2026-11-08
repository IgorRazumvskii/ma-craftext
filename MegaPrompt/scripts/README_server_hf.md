# Создание SFT-датасета на сервере с загрузкой в Hugging Face

Скрипт читает скачанные исходные Parquet, проверяет эпизоды и по мере обработки загружает части результата непосредственно в **новый** dataset-репозиторий HF. JSONL-выборка целиком на диск сервера не записывается: буферы трёх splits находятся в памяти. На диске всё равно нужны исходные траектории, временный SQLite-индекс (системный temp, либо каталог из `TMPDIR`) и служебный кэш HF. Требуется CPU; GPU не используется.

Ниже `USER@SERVER` — ваш SSH-адрес. Команды в пункте 1 выполняются на Mac, остальные — на сервере. Адрес назначения HF можно заменить на свой новый ID.

## 1. Перенос кода с Mac

Изменения режима HF пока находятся в локальных файлах, поэтому один `git clone` их не гарантирует. Передайте проверенные текущие каталоги проекта:

```bash
SERVER='USER@SERVER'
ssh "$SERVER" 'mkdir -p ~/work/ma-craftext'
rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  /Users/igor/work/ma-craftext/CrafText \
  /Users/igor/work/ma-craftext/MegaPrompt \
  /Users/igor/work/ma-craftext/MultyAgentCrafText \
  "$SERVER:~/work/ma-craftext/"
ssh "$SERVER"
```

При необходимости настройте SSH-ключ/порт в соответствии со своим сервером. Сохраните структуру `~/work/ma-craftext/{MegaPrompt,CrafText,MultyAgentCrafText}` — скрипт использует соседние файлы сценариев и декодера. После отправки новых изменений повторите `rsync`.

## 2. Окружение и доступ к HF

На сервере (Conda должна быть установлена):

```bash
conda create -n megaprompt python=3.11 -y
conda activate megaprompt
cd ~/work/ma-craftext/MegaPrompt
python -m pip install --upgrade pip
python -m pip install -e . pyarrow pillow huggingface_hub
hf auth login
hf auth whoami
```

Выберите токен с правом **write** для создаваемого dataset-репозитория и правом чтения исходного, если он приватный. Авторизация может происходить через браузер или интерактивный ввод токена. Не передавайте токен в командной строке и не присылайте его в чат.

## 3. Скачать траектории

Команда выбирает нужный каталог по его имени в репозитории, фиксирует SHA снимка и скачивает все его Parquet-файлы с сохранением путей. Источник: `igor69raz/ma-craftax-dataset`.

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo = 'igor69raz/ma-craftax-dataset'
folder = 'mappo_100k_20261101_210728_parts'
root = Path.home() / 'work/HF/datasets'
api = HfApi()
revision = api.repo_info(repo, repo_type='dataset').sha
files = api.list_repo_files(repo, repo_type='dataset', revision=revision)
parents = sorted({str(Path(f).parent) for f in files
                  if f.endswith('.parquet') and Path(f).parent.name == folder})
if len(parents) != 1:
    raise SystemExit(f'Ожидался один каталог {folder}, найдено: {parents}')
parent = parents[0]
expected = [f for f in files if str(Path(f).parent) == parent and f.endswith('.parquet')]
root.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo, repo_type='dataset', revision=revision,
                  allow_patterns=[f'{parent}/*.parquet'], local_dir=root)
parts = (root / parent).resolve()
actual = list(parts.glob('*.parquet'))
if len(actual) != len(expected):
    raise SystemExit(f'Скачано {len(actual)} файлов, ожидалось {len(expected)}')
(root.parent / 'source_parts_path.txt').write_text(str(parts) + '\n')
(root.parent / 'source_revision.txt').write_text(revision + '\n')
print(f'Снимок: {revision}\nКаталог: {parts}\nФайлов: {len(actual)}')
PY
```

Если нужного каталога в репозитории нет или их несколько, команда остановится и покажет найденные варианты. Проверить список можно через `HfApi().list_repo_files('igor69raz/ma-craftax-dataset', repo_type='dataset')`. Загрузка использует HF-кэш метаданных; при повторе `snapshot_download` обычно повторно использует уже скачанные файлы.

## 4. Запуск создания и отправки

Укажите новый, ещё не существующий dataset-репозиторий, например:

```bash
cd ~/work/ma-craftext/MegaPrompt
conda activate megaprompt

PARTS="$(cat ~/work/HF/source_parts_path.txt)"
SOURCE_REV="$(cat ~/work/HF/source_revision.txt)"
OUTPUT_REPO='igor69raz/ma-craftax-llm-sft'

python -u scripts/build_mappo_sft.py \
  --parts "$PARTS" \
  --hf-repo-id "$OUTPUT_REPO" \
  --source-hf-repo-id 'igor69raz/ma-craftax-dataset' \
  --source-hf-revision "$SOURCE_REV" \
  --skip-unreadable
```

Репозиторий создаётся **приватным**; для публичного добавьте `--hf-public`. ID должен быть свободен: существующий репозиторий скрипт не перезаписывает. По умолчанию максимальный размер JSONL-части около 64 МиБ на split; `--hf-shard-mib` меняет предел. Три буфера могут занимать около трёх таких объёмов плюс память Python. Флаг `--skip-unreadable` нужен для известной повреждённой части; пропущенные файлы будут перечислены в `report.json`. Если не хотите пропускать ошибки, уберите флаг.

Серверу нужно свободное место для исходных данных и временного индекса. Если системный `/tmp` мал, перед запуском задайте каталог на подходящем диске:

```bash
mkdir -p ~/work/tmp
export TMPDIR="$HOME/work/tmp"
```

Для долгого запуска удобно использовать `tmux`: сначала выполните `tmux new -s mappo-sft`, затем активируйте окружение и выполните команды запуска выше. Отключиться от сеанса — `Ctrl+B`, затем `D`; вернуться — `tmux attach -t mappo-sft`.

## 5. Проверка и повторный запуск

После успешного завершения откройте `https://huggingface.co/datasets/igor69raz/ma-craftax-llm-sft` (подставьте выбранный ID). Ожидаются `data/train-*.jsonl`, `data/validation-*.jsonl`, `data/test-*.jsonl`, `report.json`, `README.md`; на маленькой тестовой подвыборке часть splits может оказаться пустой. Карточка `README.md` загружается последней и служит признаком завершения скрипта. Проверяйте также код выхода процесса и статистику в отчёте.

При разрыве соединения или другой ошибке часть файлов уже может быть в репозитории. **Автоматическое продолжение не реализовано**; частичный репозиторий не считается готовым, даже если содержит данные. Для нового запуска выберите новый ID либо самостоятельно удалите частичный репозиторий в настройках HF и повторите команду. Не используйте присутствие отдельных JSONL-файлов как признак завершения.

Формат примеров и ограничения исходных данных описаны в [README_mappo_sft.md](README_mappo_sft.md). В частности, инструкции восстановлены из текущего файла сценариев, а не из отсутствующих метаданных исходного датасета.
