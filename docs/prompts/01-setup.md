# Сесія 01 — середовище, дані, каркас, перший замір швидкості

Спершу прочитай `CLAUDE.md` і `docs/SPEC.md` (розділи 5, 6, 9, 10 — уважно).
Ми відстаємо від плану приблизно на день, тому довгі завантаження запускай у фоні й паралельно роби кроки 3–4.
Питай Дмитра лише тоді, коли без нього ніяк (дозволи, вхід в акаунт, рішення з наслідками).

## Крок 0. Інвентаризація машини

- Виведи: версію Ubuntu, модель CPU, кількість ядер, обсяг RAM, відеоадаптери (`lspci`), вебкамери (`/dev/video*`), вільне місце на диску, де лежить репозиторій.
- Перевір: `git`, `uv`, `docker` (і чи працює без `sudo`), `ffmpeg`. Чого немає:
  `uv` — встанови сам офіційним інсталятором (astral.sh, у домашню теку); Python 3.12 — `uv python install 3.12`;
  `ffmpeg` та інше з apt — дай Дмитру готовий рядок `sudo apt install …` і дочекайся підтвердження.
- Kaggle CLI як інструмент (не залежність проєкту): `uv tool install kaggle`. Переконайся, що токен підхоплюється
  (наприклад, `kaggle datasets list -s violence` відпрацьовує), не виводячи значення токена.
- Hugging Face CLI як інструмент: `uv tool install "huggingface_hub[cli]"` (або `uvx`). Набори публічні — токен не потрібен.

## Крок 1. Репозиторій

- `git init`; `.gitignore`: `data/raw/`, `data/episodes/`, `weights/`, `results/`, `cache/`, `.env`, `.venv/`, `*.pt`, `*_openvino_model/`.
  `data/SOURCES.md` має комітитися.
- `pyproject.toml`: src-layout, `[project.scripts] sentinel = "sentinel.cli:main"`, runtime- і dev-залежності рівно з `CLAUDE.md`;
  `torch` і `torchvision` — з CPU-індексу PyTorch (див. `CLAUDE.md`, «Середовище»), щоб не тягнути CUDA.
- Скелет пакета за SPEC, розділ 10 (порожні модулі з docstring — лише те, що знадобиться найближчим часом), `configs/default.yaml`, `.env.example`, `README.md` (коротко: що це й як запустити).
- `uv sync`, `uv run python -c "import ultralytics, torch; print(torch.__version__, torch.cuda.is_available())"` — очікуємо версію з `+cpu` і `False`; переконайся, що в `.venv` немає пакетів `nvidia-*`.
- Коміт.

## Крок 2. Завантаження даних — у фоні

Для кожного набору — запис у `data/SOURCES.md` (URL, офіційне джерело чи дзеркало, умови використання, дата, розмір, кількість файлів).
Перед стартом оцінити обсяг і перевірити вільне місце; якщо сумарно не влазить із запасом 10 ГБ — зупинись і напиши у звіт.

1. **RWF-2000** — Hugging Face `DanJoshua/RWF-2000` → `data/raw/rwf2000/`.
   Перевір: 2000 кліпів, train 1600 / val 400, класи Fight / NonFight. Якщо відео лежать усередині parquet — розпакуй у звичайні відеофайли, зберігши поділ і класи.
2. **XD-Violence** — лише тестова частина й розмітка тесту → `data/raw/xd-violence/`.
   Спершу офіційні посилання зі сторінки https://roc-ng.github.io/XD-Violence/ (OneDrive). Якщо з терміналу не виходить — дзеркала на Kaggle або Hugging Face;
   переконайся, що є файл розмітки з інтервалами кадрів для тестових відео.
3. **Зброя на відео (Севілья)** — https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset → `data/raw/us-gun/`, за інструкцією з README.
4. **Ваги детектора зброї** — знайди 2–3 кандидати готових ваг YOLO (pistol / knife, бажано YOLOv8/YOLO11 nano або small) на GitHub, Hugging Face чи Kaggle,
   завантаж у `weights/candidates/<назва>/` разом із записом джерела, класів і ліцензії. Поки не оцінюй — лише завантаж і опиши.

## Крок 3. Мінімальний конвеєр (без ознак і агрегаторів)

- `schema.py` — pydantic-моделі опису сцени за SPEC, розділ 5 (`SceneDescription`, `Window`, `Zone`, `Summary`, `Person`, `Interaction`, `RiskResult`), `schema_version = 1`, тест серіалізації й округлення.
- `io/source.py` — `LatestFrameReader`: окремий потік читає джерело й тримає лише найсвіжіший кадр; режими `live` / `paced` / `offline`; лічильники отриманих і пропущених кадрів. Тест на `offline` зі згенерованим відео.
- `agents/detector.py` — YOLO nano (detect, клас `person`) + ByteTrack через Ultralytics (`persist=True`); повертає треки з `track_id`.
- Поза — YOLO nano pose на тих самих кадрах (окремим агентом або в `pose_features.py`, як логічніше за SPEC).

## Крок 4. `sentinel bench --source <відео>` і перший замір

- Відео: 3–5 кліпів RWF-2000 (Fight і NonFight) — або будь-яке тестове відео, якщо RWF ще завантажується.
- Для `imgsz` ∈ {640, 480, 416}: середні й p95 мс/кадр для детекції+трекінгу, пози й разом; ефективна частота аналізу; RTF за цільових 8 і 10 кадрів/с.
  Прогрів 10 кадрів не враховувати.
- Той самий замір з OpenVINO (`model.export(format="openvino")`) на CPU і, якщо iGPU вже доступна без встановлення драйверів, на iGPU (`device="intel:gpu"`) —
  разово через `uv run --with openvino`, **не додаючи** openvino в залежності.
- Таблиця → `results/bench_<дата>.md` (цей файл можна закомітити в `docs/bench/`).

## Кінець сесії

Звіт у форматі з `CLAUDE.md`. У «Цифри» — таблиця bench і розміри завантажених наборів.
У «Потрібне рішення» — твоя рекомендація: `imgsz`, частота аналізу, чи потрібен OpenVINO, і які ваги зброї варто оцінювати першими.
Далі не йди — чекай наступного завдання.
