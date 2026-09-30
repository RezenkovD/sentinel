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
uv run sentinel run --source video.mp4 --paced     # відеофайл у темпі живої камери (демонстрація)
uv run sentinel run --source video.mp4             # відеофайл без пауз (offline)
uv run sentinel run --source 0                     # вебкамера; також rtsp://…
```

`sentinel run` пише журнал вікон у `results/run_<час>.jsonl`, ключові кадри інцидентів — у `results/incidents/<id>/`.
Без `TELEGRAM_TOKEN` і `TELEGRAM_CHAT_IDS` у `.env` сповіщення йдуть у консоль і `results/incidents.jsonl`.
Демонстрацію й заміри запускати лише від мережі, у профілі `powerprofilesctl set performance`.

Заміри й перевірки:

```bash
uv run sentinel bench --source <clip>.avi [<clip>.avi …]            # люди + поза: мс/кадр, RTF
uv run sentinel bench --paced --source <clip>.avi                   # те саме в темпі живої камери
uv run sentinel bench --pipeline 480:1 640:2 crops320:2 --source …  # повний конвеєр; варіанти детектора зброї
uv run --with openvino sentinel bench --pipeline 480:1 --format openvino --source …   # те саме на OpenVINO
uv run sentinel eval build-episodes         # поділи наборів → data/splits/
uv run sentinel eval weapons                # вибір ваг детектора зброї на кадрах Севільї
uv run sentinel eval sanity --per-class 20  # санітарна перевірка конвеєра на нашому train RWF
uv run sentinel eval sanity --make-videos   # склейки для перевірки в темпі камери (потрібен ffmpeg)
uv run sentinel eval people                 # проксі «людей на кадр» для детекторів людей
```

Експеримент (`sentinel eval`) ще не реалізовано.

Налаштування (частота аналізу, розмір входу, пороги, ваги A, автомат інцидентів) — `configs/default.yaml`,
трекер — `configs/bytetrack.yaml`, зони — `configs/zones.yaml`. Журнал рішень — [`docs/DECISIONS.md`](docs/DECISIONS.md).

Ultralytics не встановлює пакети сам: `sentinel` вимикає це (`YOLO_AUTOINSTALL=false`), усі залежності —
лише з `uv.lock`; відсутні ваги не завантажуються автоматично — див. `data/SOURCES.md`.

## Розробка

```bash
uv run pytest
uv run ruff check .
```

Набори даних, ваги й результати зберігаються поза git (`data/raw/`, `weights/`, `results/`, `cache/`);
їхні джерела й умови використання описано в [`data/SOURCES.md`](data/SOURCES.md).
