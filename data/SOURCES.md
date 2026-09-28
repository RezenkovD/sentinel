# Джерела даних і ваг

Самі дані й ваги лежать поза git (`data/raw/`, `weights/`). Усі набори — лише для досліджень:
відео й кадри з людьми не поширюються й не комітяться.

## Набори даних

### RWF-2000 → `data/raw/rwf2000/`

| | |
|---|---|
| Роль | основний набір для агресії (бійка / не бійка): навчання A, тест усіх конфігурацій |
| Офіційне джерело | https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection — **відео офіційно недоступні** («Due to the privacy requirements, video files are currently not available»); запит авторам надіслано |
| Використано | **дзеркало** Hugging Face `DanJoshua/RWF-2000`: https://huggingface.co/datasets/DanJoshua/RWF-2000, файл `RWF-2000.tar.gz` (12 315 601 005 Б), останнє оновлення дзеркала 2024-08-06 |
| Умови | ліцензія авторів (README офіційного репозиторію): без дозволу SMIIP Lab не змінювати й не поширювати; не для комерційного використання; не використовувати на шкоду приватності; цитувати Cheng et al., ICPR 2020 (doi:10.1109/ICPR48806.2021.9412502). Картка дзеркала заявляє MIT — це заява третьої особи, вона не скасовує умов авторів |
| Дата завантаження | 2026-09-28; sha256 архіву `227c01e017ee3df8753388f81bf0ab65705f5009f3f0e29f7db18bacd81a2d50` — збігається з LFS-хешем HF. Клієнт `hf` (xet) зациклювався на тайм-аутах, тож архів докачано звичайним HTTP з діапазонами (curl) |
| Розмір / файли | розпаковано в `data/raw/rwf2000/{train,val}/{Fight,NonFight}/`: **2 000 кліпів AVI** — train 800 Fight + 800 NonFight, val 200 + 200 (оригінальний поділ збережено), 12 702 052 768 Б. Усі MJPEG, 30 кадрів/с, рівно 150 кадрів (5 с). Роздільність змішана (README дзеркала пише 640×360): 320×240 — 700, 1280×720 — 628, 640×360 — 169, 480×360 — 89, 1920×1080 — 31, 398×224 — 22, інші — 361. 20 кліпів `train/Fight` мають у архіві імена-«кракозябри» (китайські назви в зіпсованому кодуванні), довші за 255 байтів — межу ext4. Вони розпаковані під іменами `renamed-<перші 12 hex sha1 від UTF-8 оригінального імені><ASCII-хвіст оригіналу>`, наприклад `renamed-…_urlgot_254.avi`; відповідність — у `data/raw/rwf2000/renamed.tsv`. Мітка береться з теки, тож ім'я інформації не несе |

### XD-Violence (лише тест) → `data/raw/xd-violence/`

| | |
|---|---|
| Роль | довгі відео: затримка до тривоги, хибні тривоги на нормі (класи Fighting B1, Shooting B2) |
| Офіційне джерело | https://roc-ng.github.io/XD-Violence/ (Wu et al., ECCV 2020) |
| Використано | **офіційні** посилання: тестові відео — OneDrive (SharePoint Xidian University), архів `videos.zip` (10 804 733 011 Б), завантажено з терміналу (curl з cookie сесії публічного посилання); розмітка тесту — https://roc-ng.github.io/XD-Violence/images/annotations.txt (499 рядків: назва відео + пари «початковий / кінцевий кадр» насильницьких подій), опис міток — https://roc-ng.github.io/XD-Violence/images/ReadMe.md |
| Умови | ліцензію на сторінці не зазначено; відео зібрано з фільмів і YouTube — використовуємо лише для досліджень, не поширюємо; цитувати Wu et al., ECCV 2020 |
| Дата завантаження | 2026-09-28; цілісність архіву перевірено `unzip -t` (CRC усіх файлів — без помилок; контрольної суми OneDrive не надає) |
| Розмір / файли | розпаковано в `data/raw/xd-violence/videos/`: 800 відео MP4, 10 870 237 075 Б — 300 нормальних (`label_A`) і 500 з насильством; з них Fighting (B1) — 126, Shooting (B2) — 104 (мітки множинні). Розмітка: 500 рядків (у 14 назва має зайвий суфікс `.mp4` — при зіставленні його прибирати), 1 238 інтервалів подій; кожне насильницьке відео має розмітку |

### US Real-time gun detection in CCTV (Севільський університет) → `data/raw/us-gun/`

| | |
|---|---|
| Роль | тест сценарію «зброя» |
| Офіційне джерело | https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset → сторінка https://deepknowledge-us.github.io/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/ → Hugging Face `jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset` (посилання зі сторінки авторів, тобто офіційне) |
| Використано | «Mock attack dataset»: `weapons_images_2fps.zip` (1 485 МБ) — **кадри, розмічені з частотою 2 кадри/с** з камер Cam1 (607), Cam5 (1031), Cam7 (3511), рамки зброї. Відеофайлів у наборі немає. Синтетичний Unity-набір (U0.5 / U1 / U2.5 / U5) не завантажено — він потрібен лише для донавчання детектора |
| Умови | CC BY-NC 4.0; безкоштовно для академічних досліджень із цитуванням Salazar González et al., Neural Networks, 2020 (doi:10.1016/j.neunet.2020.09.013); комерційне використання — лише за згодою авторів |
| Дата завантаження | 2026-09-28 (`weapons_images_2fps.zip`, 1 485 019 798 Б, sha256 `558f5baa109abe8fba57b61fa177c75358001f5499f5e78178f4bf7dc9f55e8c` — збігається з LFS-хешем HF) |
| Розмір / файли | розпаковано в `data/raw/us-gun/Images/`: 5 149 кадрів JPG 1920×1080 + 5 149 розміток Pascal VOC XML, 1 489 743 767 Б. Імена: `Cam<N>-…_Segment_<k>_x264_frame_<i>` — кадри одного сегмента утворюють послідовність 2 кадри/с. Об'єкти: Handgun 1 714, Short_rifle 797, Knife 210; кадрів без зброї 3 615 (70 %), зі зброєю 1 534 |

### Допоміжне тестове відео → `data/raw/vtest.avi`

| | |
|---|---|
| Роль | лише димовий тест детектора й трекера, поки RWF-2000 завантажувався; в експеримент не входить |
| Джерело | зразок OpenCV: https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi (репозиторій OpenCV, Apache 2.0) |
| Дата / розмір | 2026-09-28; 8 131 690 Б; 768×576, 10 кадрів/с, 795 кадрів (79,5 с) |

## Ваги

### Люди й поза (Ultralytics) → `weights/`

Джерело — офіційні релізи Ultralytics: `https://github.com/ultralytics/assets/releases/download/v8.4.0/<файл>`.
Ліцензія — AGPL-3.0 (Ultralytics). Завантажено 2026-09-28. Навчені на COCO (detect: 80 класів, використовуємо `person`; pose: 17 ключових точок).

| Файл | Розмір, Б | sha256 |
|---|---|---|
| `yolo11n.pt` | 5 613 764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| `yolo11n-pose.pt` | 6 255 593 | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` |
| `yolo26n.pt` | 5 544 453 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `yolo26n-pose.pt` | 7 878 574 | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |

### Детектор зброї — кандидати → `weights/candidates/<назва>/`

Завантажено 2026-09-28, ще не оцінювалися. Метрики — **заявлені авторами на їхньому власному поділі**, нами не перевірені.
Докладно (URL із закріпленим комітом, train_args, дані, умови) — у `weights/candidates/<назва>/SOURCE.md`.
Ваги не розпаковувались (не виконувався `torch.load`): статичний аналіз pickle (`pickletools`) — посилання лише на
`torch.*`, `ultralytics.*`, `collections.OrderedDict`, `__builtin__.set`. Кожен чекпойнт Ultralytics містить
автоматичне поле `license: AGPL-3.0`, що суперечить заявленій авторами MIT.

| Назва | Архітектура | Класи | Дані навчання | Заявлені метрики | Ліцензія ваг / даних | Розмір, Б | sha256 |
|---|---|---|---|---|---|---|---|
| `hf-cosgun99-gun-knife-yolo11n` — https://huggingface.co/cosgun99/gun-knife-yolo11n (офіційне завантаження автора) | YOLO11n, 640, Ultralytics 8.4.46 | `gun`, `knife` | «custom merged», джерела не названо; 29 446 / 3 681 / 3 681 зображень | test mAP50 0,964, mAP50-95 0,698 | MIT (тег HF) / не зазначено | 5 474 259 | `07241a8fc57d0f3fd29231dd72b9f452375745290b2185440de0165590997cb5` |
| `gh-halok600-weaponshield-yolov8s` — https://github.com/Halok600/WeaponShield-AI (репозиторій автора) | YOLOv8s, 640, Ultralytics 8.4.104 | `Pistol`, `Rifle`, `Knife` | ≈31,6 тис. зображень: два набори Roboflow Universe + 5 000 фонових COCO val2017 + 559 перерозмічених Kaggle | test mAP50 0,833, mAP50-95 0,613 (Pistol 0,908 / 0,691; Knife 0,636 / 0,405) | MIT (файл LICENSE) / Roboflow — CC BY 4.0 (не підтверджено, сторінки недоступні) | 22 489 763 | `0a96416a2f22cec4a3cc0232747a4ed98edb6fa6b4deb8978dd8c7ba7d944934` |
| `hf-subh775-threat-detection-yolov8n` — https://huggingface.co/Subh775/Threat-Detection-YOLOv8n (офіційне завантаження автора) | YOLOv8n, 640, Ultralytics 8.3.205 | `Gun`, `explosion`, `grenade`, `knife` | неназваний набір Roboflow (Kaggle `threat-detection-3849`), ≈1,1 тис. анотацій Gun і ≈1,0 тис. knife | val mAP50 0,813, mAP50-95 0,553 | MIT (тег HF) / не зазначено | 6 251 747 | `86c43444ae8319d2276dd300edc3e7f7a1137fe7566737f994ca579ad770f6ce` |

Знайдено, але не завантажено: `0207yjy/Weapon_Detection_YOLO26n` (GitHub; YOLO26n 320 px на академічному наборі SOHAS,
pistol / knife + схожі предмети, заявлено val mAP50 0,948; у репозиторії немає ліцензії) — найсильніший запасний кандидат;
Kaggle `killa92/map-0-9-guns-and-knifes-detection-yolov11` — потребує Kaggle-токена.
Пастка: `Tunahan445/Gun_and_Knife_dedection_YOLO8n` побайтово збігається зі стандартним COCO `yolov8n.pt` — це не детектор зброї.
