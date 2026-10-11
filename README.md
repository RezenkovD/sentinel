# Sentinel

Мультиагентна система оцінки ситуаційної небезпеки на відео з камер спостереження. Сценарії — **зброя в руці** та
**агресивна поведінка (бійка)**. Програмний продукт магістерської роботи (КПІ ім. Ігоря Сікорського, 2026).

```
відео (вебкамера / RTSP / файл)
  → агенти ознак: люди, зброя, супровід, поза й взаємодія, контекст зони
  → опис сцени (pydantic → JSON) раз на ~1 с
  → агрегатор A (логістична регресія)  і/або  агрегатор B (LLM, лише текстовий опис)  → каскад (B лише за потреби)
  → автомат інцидентів (гістерезис; «зброя в руці» двічі поспіль — тривога одразу)
  → одне повідомлення в Telegram на інцидент: фото з розмитими обличчями, далі лише редагується
```

Наукова частина — чесне порівняння базових методів, агрегатора A, агрегатора B і каскаду на **ідентичних** описах
сцени. Специфікація — [`docs/SPEC.md`](docs/SPEC.md), журнал рішень — [`docs/DECISIONS.md`](docs/DECISIONS.md),
звіти етапів — [`docs/reports/`](docs/reports/), заміри — [`docs/bench/`](docs/bench/).

Що свідомо **не** робиться: зображення в LLM не передаються (лише текстовий опис сцени); обличчя не розпізнаються,
а на кожному кадрі, що залишає машину, розмиваються; сценаріїв поза «зброя + агресія» немає.

## Що є в репозиторії, а чого немає

Є: код (`src/sentinel/`), конфіги (`configs/`), тести (`tests/`), Dockerfile, скрипти (`scripts/`), документація
й поділи наборів на train / val / test (`data/splits/`: ідентифікатори кліпів, seed і правила поділу, без кадрів).

**Немає** — і не буде, бо умови наборів дозволяють лише дослідницьке використання без поширення:
- наборів даних (RWF-2000, XD-Violence, US Real-time gun detection in CCTV) і будь-яких відео чи кадрів із людьми;
- ваг моделей (`weights/`), зокрема навчених коефіцієнтів агрегатора A;
- результатів, журналів і кешу відповідей LLM (`results/`, `cache/`);
- секретів: токен бота, id чатів і ключі LLM живуть лише в локальному `.env`.

Звідки брати набори й ваги, їхні ліцензії, розміри й SHA-256 — [`data/SOURCES.md`](data/SOURCES.md).

## Вимоги

- Linux (розроблено й перевірено на Ubuntu), Python 3.12, [uv](https://docs.astral.sh/uv/).
- Лише CPU: PyTorch ставиться в CPU-збірці (без CUDA), моделі за замовчуванням працюють через OpenVINO.
- `ffmpeg` / `ffprobe` — для демо-відео й склейок; Docker — за бажанням; вебкамера — для живого запуску.
- За бажанням: бот Telegram (сповіщення й команди) і ключ OpenAI-сумісного API для агрегатора B
  (використовується безкоштовний рівень Groq, модель `openai/gpt-oss-120b`).

## Встановлення

```bash
uv sync                    # .venv точно за uv.lock (torch / torchvision — з CPU-індексу PyTorch)
cp .env.example .env       # заповнити за потреби; .env у git не потрапляє
```

**Ваги** покладіть у `weights/` за [`data/SOURCES.md`](data/SOURCES.md) (шляхи — у `configs/default.yaml`):

| Що | Файл | Звідки |
|---|---|---|
| Люди | `weights/yolo11n.pt` | офіційні релізи Ultralytics |
| Поза (17 ключових точок) | `weights/yolo26n-pose.pt` | офіційні релізи Ultralytics |
| Зброя | `weights/candidates/gh-halok600-weaponshield-yolov8s/best.pt` | WeaponShield YOLOv8s (репозиторій автора) |
| Агрегатор A | `weights/aggregator_a/a_v1_1_all.json` | навчається з кешу описів сцени: `uv run sentinel eval train-a` |

Ultralytics нічого не завантажує й не встановлює сам (`YOLO_AUTOINSTALL=false`): відсутні ваги — це помилка з
підказкою. Без навченого A можна прибрати `aggregator_a.model` з конфігу — тоді діють задані вручну ваги v0.1.

Разовий експорт моделей в OpenVINO (перегенеровується, коли змінюються ваги):

```bash
uv run sentinel export
```

## Запуск

```bash
uv run sentinel run --source 0                          # вебкамера (індекс пристрою)
uv run sentinel run --source rtsp://<камера>/stream     # RTSP-потік
uv run sentinel run --source video.mp4                  # файл: детерміновано, кадр за кадром, без пауз
uv run sentinel run --source video.mp4 --paced          # файл у темпі живої камери
scripts/live_view.sh                                    # у другому терміналі: рядок на кожне вікно аналізу
```

Читач завжди бере найсвіжіший кадр (черг кадрів немає), аналіз — з фіксованою частотою (`analysis.fps`), опис
сцени — раз на вікно. `sentinel run` пише журнал вікон у `results/run_<час>.jsonl`, ключові кадри інцидентів — у
`results/incidents/<id>/`; наприкінці друкує підсумок (кадри, вікна, інциденти, RTF, частка пропущених кадрів).

**Telegram.** З `TELEGRAM_TOKEN` і `TELEGRAM_CHAT_IDS` у `.env` на кожен інцидент приходить одне повідомлення з
кадром, яке потім редагується («✅ Завершено …»). Бот приймає команди лише з дозволених чатів:
`/status`, `/mode A | B | cascade`, `/mute 10m`. Без токена сповіщення друкуються в консоль і пишуться в
`results/incidents.jsonl`.

**Режими агрегації** — `aggregation.mode` у конфігу або `/mode` наживо:
- `A` — логістична регресія на 17 ознаках опису сцени, мілісекунди;
- `B` — LLM на кожне вікно (потрібен `GROQ_API_KEY`);
- `cascade` — A завжди, B — у сірій зоні ризику A (0,50–0,55, обрано на val), на вікнах УВАГИ й ЗАТИХАННЯ і коли
  ризик A змінив би стан інциденту.

B ніколи не гальмує конвеєр: вікно бере відповідь B, якщо вона вже прийшла, інакше рішення ухвалює A. Ліміти
провайдера (`llm.providers` у конфігу) не обходяться: на безкоштовному рівні Groq (8 000 токенів/хв) це ≈ 4 відповіді B на хвилину.

Налаштування — `configs/default.yaml` (частоти, розміри входу, пороги автомата, сіра зона), трекер —
`configs/bytetrack.yaml`, профілі зон — `configs/zones.yaml`. Для замірів швидкості — живлення від мережі й
`powerprofilesctl set performance`.

## Docker

Образ лише CPU; ваги, результати й `.env` монтуються під час запуску і в образ не потрапляють.

```bash
docker build -t sentinel:dev .
docker run --rm -v "$PWD/weights:/app/weights:ro" --entrypoint pytest sentinel:dev -q      # тести в чистому середовищі
docker run --rm --user "$(id -u):$(id -g)" --env-file .env --device /dev/video0 --group-add video \
  -v "$PWD/weights:/app/weights:ro" -v "$PWD/results:/app/results" sentinel:dev run --source 0   # вебкамера
```

Часовий пояс образу — `Europe/Kyiv` (`-e TZ=…`, щоб змінити). Експорт OpenVINO має лежати в `weights/`
(`sentinel export` на хості або в контейнері з `weights` без `:ro`).

## Тести

```bash
uv run pytest -q          # схема, ознаки на синтетичних ключових точках, A, B (без мережі), каскад, автомат, бот
uv run ruff check .
```

У тестах моделі не завантажуються й мережа не використовується.

## Відтворення експерименту

Потрібні набори даних (див. `data/SOURCES.md`); усе рахується на CPU, детерміновано (фіксовані seed, кеш LLM).

```bash
uv run sentinel eval build-episodes         # поділи наборів → data/splits/
uv run sentinel eval extract --plan all     # кеш описів сцени (один прогін сприйняття на кліп; відновлюваний)
uv run sentinel eval feature-auc            # одновимірні AUC ознак
uv run sentinel eval train-a                # A: навчання на train, оцінка епізоду й поріг — на val
uv run sentinel eval fsm-tune               # θ_low і M автомата на склейках val
uv run sentinel eval baselines              # пороги базових методів (рух, детектор зброї)
uv run sentinel eval b-run --clips 120      # B на 120 кліпах val (безкоштовний Groq, відновлюване)
uv run sentinel eval b-val                  # B проти A на тих самих вікнах val
uv run sentinel eval cascade-grid           # крива «частка вікон до B — AUC епізодів» для сірої зони
uv run sentinel eval experiment freeze      # тест відкривається один раз: хеші A, конфігу, кешу, порогів
uv run sentinel eval experiment rwf|sevilla|xd|report   # тест: 95 % ДІ бутстрепом, ROC
scripts/protocol.sh --build                 # протокол випробувань у Docker → docs/reports/protocol_<дата>.md
uv run sentinel bench --source <clip>.avi   # швидкість: мс/кадр, RTF
```

Пороги й гіперпараметри підбираються лише на val; тест відкрито один раз після заморожування
([`docs/experiment_freeze.json`](docs/experiment_freeze.json)).

## Ліцензія

[GNU AGPL-3.0](LICENSE). Причина — Ultralytics (YOLO, детекція, поза, ByteTrack), на якій побудовано сприйняття:
бібліотека й офіційні ваги поширюються під AGPL-3.0, тож і похідна система, зокрема доступна по мережі,
відкривається на тих самих умовах. Ваги детектора зброї (WeaponShield YOLOv8s) автор позначає як MIT, але чекпойнт
Ultralytics записує AGPL-3.0 — трактуємо як AGPL-3.0 (`data/SOURCES.md`).

Набори даних мають власні умови (лише дослідницьке використання) і в репозиторій не входять.
