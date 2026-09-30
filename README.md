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

Бекенд моделей за замовчуванням — OpenVINO (`models.backend: openvino`, рішення 30.09). Він потребує разового
експорту, який перегенеровується, коли змінюються ваги; без нього `sentinel run` зупиниться з підказкою:

```bash
uv run sentinel export
```

Заміри й перевірки:

```bash
uv run sentinel bench --source <clip>.avi [<clip>.avi …]            # люди + поза: мс/кадр, RTF
uv run sentinel bench --paced --source <clip>.avi                   # те саме в темпі живої камери
uv run sentinel bench --pipeline 480:1 640:2 crops320:2 --source …  # повний конвеєр; варіанти детектора зброї
uv run sentinel bench --pipeline 480:2 --format openvino --source …   # те саме на OpenVINO
uv run sentinel bench --pipeline 480:2 640:2 --quiet --people detect:weights/yolo11n.pt --backends openvino torch --source …
                                            # чекає тихого вікна, шумні прогони повторює (сесія 03)
uv run sentinel eval build-episodes         # поділи наборів → data/splits/
uv run sentinel eval weapons                # вибір ваг детектора зброї на кадрах Севільї
uv run sentinel eval sanity --per-class 20  # санітарна перевірка конвеєра на нашому train RWF
uv run sentinel eval sanity --make-videos   # склейки для перевірки в темпі камери (потрібен ffmpeg)
uv run sentinel eval people                 # проксі «людей на кадр» для джерел людей
uv run sentinel eval xd-subset              # підвибірка тесту XD-Violence → data/splits/xd_subset.json
uv run sentinel eval extract --plan all     # кеш описів сцени: RWF train+val → test → Севілья → XD (відновлюване)
uv run sentinel eval feature-auc            # одновимірні AUC ознак A на кеші
uv run sentinel eval train-a                # навчання A (train), вибір C, оцінки епізоду й порогів (val)
uv run sentinel eval weapon-rule            # правило «зброя в руці» на кеші: Севілья dev і NonFight train
uv run sentinel eval baselines              # пороги baseline_weapon і baseline_motion
```

Кеш описів сцени — `data/episodes/<набір>/<епізод>.jsonl` (поза git): перший рядок — метадані (версії схеми й
сприйняття, хеш налаштувань), далі рядок на вікно: опис сцени й службове `aux`, якого агрегатори не бачать.
Повний експеримент (`sentinel eval` для A / B / каскаду) — наступні сесії.

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
