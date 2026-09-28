# Sentinel

Мультиагентна система оцінки ситуаційної небезпеки на відео з камер спостереження (сценарії: зброя та агресивна поведінка).
Програмний продукт переддипломної практики й магістерської роботи (КПІ ім. Ігоря Сікорського).

Відео (вебкамера / RTSP / файл) → агенти ознак (люди, зброя, трекінг, поза, контекст зони) → структурований **опис сцени**
раз на ~1 с → агрегатор A (логістична регресія + правило «зброя в руці») і/або агрегатор B (LLM, лише текст)
→ автомат інцидентів → одне повідомлення в Telegram на інцидент.

Повна специфікація — [`docs/SPEC.md`](docs/SPEC.md).

## Встановлення

Потрібні Linux, Python 3.12 і [uv](https://docs.astral.sh/uv/). PyTorch ставиться лише в CPU-збірці (без CUDA).

```bash
uv sync                 # створює .venv за uv.lock
cp .env.example .env    # секрети (токен бота, ключ LLM) — лише тут
```

## Запуск

```bash
uv run --with lap sentinel bench --source data/raw/rwf2000/val/Fight/<clip>.avi   # замір швидкості (RTF, мс/кадр)
uv run --with lap sentinel bench --paced --source <clip>.avi                     # те саме в темпі живої камери
uv run sentinel run --source 0                                          # вебкамера (ще не реалізовано)
uv run sentinel run --source video.mp4 --paced                          # відеофайл у темпі живої камери (ще не реалізовано)
uv run sentinel eval --config all                                       # експеримент (ще не реалізовано)
```

Налаштування (частота аналізу, розмір входу, пороги) — `configs/default.yaml`, трекер — `configs/bytetrack.yaml`.

`--with lap`: трекер ByteTrack в Ultralytics потребує пакет `lap`, якого поки немає в залежностях проєкту
(чекає рішення; до того `lap` береться тимчасово й без фіксації версії). Ultralytics не встановлює пакети сам:
`sentinel` вимикає це (`YOLO_AUTOINSTALL=false`), а відсутні ваги не завантажуються автоматично — див. `data/SOURCES.md`.

## Розробка

```bash
uv run pytest
uv run ruff check .
```

Набори даних, ваги й результати зберігаються поза git (`data/raw/`, `weights/`, `results/`, `cache/`);
їхні джерела й умови використання описано в [`data/SOURCES.md`](data/SOURCES.md).
