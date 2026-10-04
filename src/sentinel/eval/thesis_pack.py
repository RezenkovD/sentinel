"""Thesis pack (session 06, step 4): every key number, table, figure, screen and listing in one archive.

    uv run --with matplotlib sentinel eval thesis-pack      # -> results/thesis_pack_<date>.zip

The texts (thesis, report, paper, slides) are written elsewhere from this pack, so no number is retyped by
hand: `numbers.md` gives each with the file it comes from and that file's date. matplotlib is not a project
dependency — it comes with `uv run --with matplotlib`; without it the figures are skipped (and listed as
missing). The archive holds frames with people from research datasets: it stays out of git and is not shared.
Rebuild it with the same command whenever new results land (B on val, the cascade grid).
"""

import argparse
import csv
import inspect
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from sentinel.config import DEFAULT_CONFIG, Config, load_config

RESULTS = Path("results")
SPEED_HEAD = [
    "Люди", "Зброя", "Бекенд", "мс/кадр, сер.", "мс/кадр, p95", "RTF",
    "Кадр/с у темпі камери", "Затримка p95, мс",
]  # fmt: skip
EXPERIMENT = RESULTS / "experiment"


# --- numbers --------------------------------------------------------------------------------------------


def _date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%d.%m.%Y") if path.exists() else "—"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _pct(share: float) -> str:
    return f"{100 * share:.1f}".replace(".", ",") + " %"


def _n(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}".replace(".", ",")


def _ci(value, ci, digits: int = 3) -> str:
    return _n(value, digits) + (f" [{_n(ci[0], digits)}; {_n(ci[1], digits)}]" if ci else "")


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class Numbers:
    """Markdown sections of `- name: value — source (date)` lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def section(self, title: str) -> None:
        self.lines += ["", f"## {title}", ""]

    def add(self, name: str, value: str, source: Path | str) -> None:
        when = _date(source) if isinstance(source, Path) else ""
        tail = f"`{source}`" + (f", {when}" if when else "")
        self.lines.append(f"- {name}: **{value}** — {tail}")

    def text(self, title: str) -> str:
        return "\n".join([f"# {title}", *self.lines]) + "\n"


def machine(numbers: Numbers) -> dict:
    from sentinel.eval.protocol import environment

    config = load_config(DEFAULT_CONFIG)
    env = environment(config)
    mem = next(
        (line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal")), ""
    )
    ram_gb = int(re.findall(r"\d+", mem)[0]) / 1024**2 if mem else None
    numbers.section("Машина й середовище")
    numbers.add(
        "Процесор", f"{env['cpu']}, {env['cpus']} логічних ядер; GPU не використовується", "/proc/cpuinfo"
    )
    numbers.add("Оперативна пам'ять", f"{_n(ram_gb, 1)} ГБ", "/proc/meminfo")
    numbers.add("ОС", platform.platform(), "platform")
    numbers.add(
        "Python і бібліотеки",
        f"Python {env['python']}; " + ", ".join(f"{k} {v}" for k, v in env["versions"].items()),
        "середовище uv",
    )
    numbers.add("Бекенд моделей", env["backend"], DEFAULT_CONFIG)
    return env


def speed(numbers: Numbers) -> list[dict]:
    path = RESULTS / "bench_quiet_2026-10-03.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    numbers.section("Швидкість (тихий замір, від мережі, профіль performance, 10 кадрів/с, 20 кліпів)")
    for r in rows:
        numbers.add(
            f"{r['people']}, зброя {r['weapons']}, {r['backend']}",
            f"RTF {_n(r['rtf'], 2)}; {_n(r['total_ms']['mean_ms'], 1)} / {_n(r['total_ms']['p95_ms'], 1)} "
            "мс/кадр "
            f"(сер. / p95); paced {_n(r['paced']['effective_fps'], 1)} кадр/с, затримка p95 "
            f"{_n(r['paced']['latency_ms']['p95_ms'], 0)} мс",
            path,
        )
    for name in ("bench_crowded6_2026-10-04.jsonl", "bench_crowded_2026-10-04.jsonl"):
        p = RESULTS / name
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                numbers.add(
                    f"людна сцена ({'6–8' if 'crowded6' in name else '12–20'} людей), зброя {r['weapons']}",
                    f"RTF {_n(r['rtf'], 2)}; {_n(r['total_ms']['mean_ms'], 1)} мс/кадр",
                    p,
                )
    protocol = sorted((p for p in RESULTS.glob("protocol_*.json") if "_trial" not in p.name), key=_mtime)
    if protocol:
        clips = json.loads(protocol[-1].read_text())["clips"]
        numbers.add(
            "Протокол випробувань у Docker (paced)",
            f"RTF {_n(min(c['rtf'] for c in clips), 2)}–{_n(max(c['rtf'] for c in clips), 2)}; "
            "затримка p95 до "
            f"{max(c['latency_p95_ms'] for c in clips):.0f} мс; {len(clips)} кліпів",
            protocol[-1],
        )
    return rows


def datasets(numbers: Numbers, episodes: Path = Path("data/episodes")) -> list[list[str]]:
    table = []
    numbers.section("Набори даних і поділи")
    for dataset, title in (("rwf2000", "RWF-2000"), ("us_gun", "Севілья (зброя)"), ("xd", "XD-Violence")):
        parts: dict[str, list[float]] = {}
        for path in sorted((episodes / dataset).rglob("*.jsonl")):
            with path.open(encoding="utf-8") as f:
                meta = json.loads(f.readline())["meta"]
            parts.setdefault(meta["part"], []).append(meta)
        for part, metas in sorted(parts.items()):
            hours = sum(m["duration_s"] for m in metas) / 3600
            windows = sum(m["windows"] for m in metas)
            positives = sum(int(m["label"]) for m in metas)
            row = [title, part, str(len(metas)), str(positives), str(windows), _n(hours, 2)]
            table.append(row)
            numbers.add(
                f"{title}, {part}",
                f"{len(metas)} епізодів ({positives} з подією), {windows} вікон, {_n(hours, 2)} год",
                episodes / dataset,
            )
    numbers.add(
        "Поділ RWF",
        "test = оригінальний val; наш train / val = 80 / 20 за відео-джерелами",
        Path("data/splits/rwf2000.json"),
    )
    numbers.add("Поділ Севільї", "dev — Cam1 + Cam5, test — Cam7", Path("data/splits/us_gun.json"))
    numbers.add(
        "Підвибірка XD", "156 відео: B1 48, B2 77, норма 31 (seed 2026)", Path("data/splits/xd_subset.json")
    )
    return table


def quality(numbers: Numbers) -> None:
    a_val = _json(RESULTS / "aggregator_a_v1_1_all_2026-10-03.json")
    base_val = _json(RESULTS / "baselines_2026-09-30.json")
    rwf = _json(EXPERIMENT / "rwf_test.json")
    numbers.section("Агрегатор A v1.1 і базові методи — RWF-2000")
    if a_val:
        m = a_val["metrics_val"]
        source = RESULTS / "aggregator_a_v1_1_all_2026-10-03.json"
        numbers.add(
            "A v1.1, val: AUC епізодів / вікон",
            f"{_n(m['episode_auc_val'])} / {_n(m['window_auc_val_all'])}",
            source,
        )
        numbers.add("A v1.1, val: F1 за порогом 0,45", _n(m["episode_f1_val"]), source)
        numbers.add(
            "A v1.1: регуляризація C, вікон навчання", f"{a_val['c']}, {a_val['training_windows']}", source
        )
    if base_val:
        source = RESULTS / "baselines_2026-09-30.json"
        mo = base_val["motion"]
        numbers.add(
            "Рух (базовий), val: AUC епізодів, F1, τ",
            f"{_n(mo['episode_auc'])}, {_n(mo['f1'])}, τ {_n(mo['threshold'], 2)}",
            source,
        )
    if rwf:
        source = EXPERIMENT / "rwf_test.json"
        for name, title in (
            ("A", "A v1.1"),
            ("baseline_motion", "Рух (базовий)"),
            ("baseline_weapon", "Зброя (базовий)"),
        ):
            r = rwf["configs"][name]
            c = r["confusion"]
            numbers.add(
                f"{title}, test: AUC епізодів [95 % ДІ]; F1; точність; повнота",
                f"{_ci(r['auc'], r['auc_ci'])}; {_n(r['f1'])}; {_n(r['precision'])}; {_n(r['recall'])} "
                f"(TP {c['tp']} / FP {c['fp']} / FN {c['fn']} / TN {c['tn']})",
                source,
            )
        for base, d in rwf.get("a_minus_baseline_auc", {}).items():
            numbers.add(
                f"A − {base}, test: різниця AUC [парний бутстреп]", _ci(d["delta_auc"], d["ci"]), source
            )
    numbers.section("Детектор зброї й правило «зброя в руці»")
    rule = _json(RESULTS / "weapon_rule_2026-09-30.json")
    if rule:
        r = rule["кадр 480:2"]["confirm2"]
        numbers.add(
            "Dev (Севілья Cam1 + Cam5), кадр 480:2, 2 вікна поспіль: повнота сегментів; медіана часу; "
            "NonFight train з тривогою",
            f"{_n(r['recall'])}; {_n(r['median_time_to_alarm_s'], 1)} с; "
            f"{_pct(r['rwf_nonfight_false_alarm_share'])}",
            RESULTS / "weapon_rule_2026-09-30.json",
        )
    sweep = _json(RESULTS / "weapon_crops_sweep_2026-10-03.json")
    if sweep:
        r = next(x for x in sweep["rows"] if x["tau"] == 0.5)
        numbers.add(
            "Dev, вирізки 320:2, поріг 0,50 (не обрано — швидкість)",
            f"повнота {_n(r['recall'])}; {_n(r['median_time_to_alarm_s'], 1)} с; "
            f"хибних {_pct(r['rwf_nonfight_false_alarm_share'])}",
            RESULTS / "weapon_crops_sweep_2026-10-03.json",
        )
    sev = _json(EXPERIMENT / "sevilla_test.json")
    if sev:
        for key, title in (("rule", "правило"), ("baseline_weapon", "детектор без підтвердження")):
            r = sev["configs"][key]
            numbers.add(
                f"Test (Cam7), {title}: повнота сегментів; медіана часу; хибних на год; "
                "RWF test NonFight з тривогою",
                f"{_ci(r['recall'], r['recall_ci'], 2)}; {_n(r['median_time_to_alarm_s'], 1)} с; "
                f"{_n(r['false_per_h'], 1)}; {_pct(r['rwf_nonfight_alarm']['share'])}",
                EXPERIMENT / "sevilla_test.json",
            )
    xd = _json(EXPERIMENT / "xd_test.json")
    if xd:
        numbers.section("XD-Violence (156 відео, перенесення на інший домен)")
        for name, title in (("A", "A v1.1"), ("baseline_motion", "Рух"), ("baseline_weapon", "Зброя")):
            w = xd["configs"][name]["window"]
            al = xd["configs"][name]["alarms"]
            numbers.add(
                f"{title}: AUC вікон усі / B1 / B2",
                " / ".join(_ci(w[g]["auc"], w[g]["auc_ci"]) for g in ("all", "B1", "B2")),
                EXPERIMENT / "xd_test.json",
            )
            numbers.add(
                f"{title}: сегментів з тривогою; медіана затримки; хибних на год (усі / лише норма)",
                f"{_n(al['all']['detected'], 2)}; {_n(al['all']['median_delay_s'], 1)} с; "
                f"{_n(al['all']['false_per_h'], 1)} / {_n(al['normal']['false_per_h'], 1)}",
                EXPERIMENT / "xd_test.json",
            )


def automaton(numbers: Numbers, config: Config) -> None:
    inc = config.incidents
    numbers.section("Автомат інцидентів")
    numbers.add(
        "Параметри",
        f"θ_high {_n(inc.theta_high, 2)}, θ_low {_n(inc.theta_low, 2)}, k {inc.k} з n {inc.n}, "
        f"M {_n(inc.quiet_s, 0)} с, "
        f"злиття {_n(inc.merge_after_close_s, 0)} с, зброя — {inc.weapon_confirm_windows} вікна поспіль",
        DEFAULT_CONFIG,
    )
    fsm = _json(RESULTS / "fsm_tuning_2026-09-30.json")
    if fsm:
        b = fsm["best"]
        numbers.add(
            "Підбір θ_low і M на склейках val: опрацьовано бійок; хибних інцидентів на год",
            f"{100 * b['handled']:.0f} %; {_n(b['calm_incidents_per_h'], 1)}",
            RESULTS / "fsm_tuning_2026-09-30.json",
        )


def aggregator_b(numbers: Numbers) -> None:
    numbers.section("Агрегатор B (LLM) — пілот і стан на val")
    pilot = _json(RESULTS / "b_pilot_b_v1.json")
    if pilot:
        m = pilot["metrics"]
        source = RESULTS / "b_pilot_b_v1.json"
        numbers.add(
            "Пілот b_v1: вікон, валідних відповідей",
            f"{m['windows']}, {100 * m['valid_share']:.0f} %",
            source,
        )
        numbers.add(
            "Пілот: AUC вікон B / A на тих самих вікнах",
            f"{_n(m['window_auc_b'])} / {_n(m['window_auc_a_same_windows'])}",
            source,
        )
        numbers.add(
            "Пілот: затримка p50 / p95",
            f"{_n(m['latency_p50_s'], 2)} / {_n(m['latency_p95_s'], 2)} с",
            source,
        )
        t = m["tokens_mean"]
        numbers.add(
            "Пілот: токенів на виклик (вхід / вихід / міркування)",
            f"{t['input']:.0f} / {t['output']:.0f} / {t['reasoning']:.0f}",
            source,
        )
        c = m["consistency"]
        numbers.add(
            "Узгодженість повторів (40 вікон): σ ризику, збіг рівня",
            f"{_n(c['risk_std_mean'])}, {100 * c['level_agreement']:.0f} %",
            source,
        )
    log = RESULTS / "b_run_val120.log"
    if log.exists():
        last = [line for line in log.read_text().splitlines() if "windows of" in line]
        numbers.add("Прогін B на 120 кліпах val (безкоштовний Groq)", last[-1] if last else "—", log)
    numbers.add(
        "Бюджет B на всьому експерименті (тариф Groq)",
        "$6,25 (B на кожному вікні); безкоштовно — ≈ 148 діб; через OpenRouter / DeepInfra ≈ $1,62",
        Path("docs/reports/05-aggregator-B-cascade.md"),
    )
    grid = sorted(RESULTS.glob("cascade_grid_*.json"))
    if grid:
        numbers.add(
            "Каскад: обрана сіра зона", json.dumps(_json(grid[-1]).get("best"), ensure_ascii=False), grid[-1]
        )


def code_size(numbers: Numbers) -> None:
    def lines(root: str) -> int:
        return sum(
            sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
            for p in Path(root).rglob("*.py")
        )

    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
    )
    collected = sum(1 for line in done.stdout.splitlines() if "::" in line)
    numbers.section("Код")
    numbers.add("Автоматичних тестів (pytest)", str(collected), "pytest --collect-only")
    numbers.add("Непорожніх рядків коду: src / tests", f"{lines('src')} / {lines('tests')}", "src/, tests/")


# --- tables ---------------------------------------------------------------------------------------------


def write_table(folder: Path, name: str, title: str, head: list[str], rows: list[list[str]]) -> None:
    from sentinel.eval.experiment import markdown_table

    (folder / f"{name}.md").write_text(
        "\n".join([f"**{title}**", "", *markdown_table(head, rows)]) + "\n", encoding="utf-8"
    )
    with (folder / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(head)
        writer.writerows(rows)


def tables(folder: Path, speed_rows: list[dict], dataset_rows: list[list[str]]) -> None:
    from sentinel.eval.experiment import tables as experiment_tables

    folder.mkdir(parents=True, exist_ok=True)
    for name, (title, head, rows) in experiment_tables(EXPERIMENT).items():
        write_table(folder, f"test_{name}", title, head, rows)
    write_table(
        folder,
        "speed",
        "Таблиця. Швидкість повного конвеєра на CPU (10 кадрів/с)",
        SPEED_HEAD,
        [
            [
                r["people"], r["weapons"], r["backend"],
                _n(r["total_ms"]["mean_ms"], 1), _n(r["total_ms"]["p95_ms"], 1), _n(r["rtf"], 2),
                _n(r["paced"]["effective_fps"], 1), _n(r["paced"]["latency_ms"]["p95_ms"], 0),
            ]
            for r in speed_rows
        ],
    )  # fmt: skip
    write_table(
        folder,
        "datasets",
        "Таблиця. Набори даних і поділи",
        ["Набір", "Частина", "Епізодів", "З подією", "Вікон (1 с)", "Годин"],
        dataset_rows,
    )


# --- figures --------------------------------------------------------------------------------------------


def figures(folder: Path) -> list[str]:
    """PNG, 300 dpi, white background, Ukrainian labels (DejaVu Sans); returns the ones not made."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["усі (немає matplotlib — запускати через `uv run --with matplotlib`)"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )
    folder.mkdir(parents=True, exist_ok=True)
    missing = []
    names = {"A": "Агрегатор A", "baseline_motion": "Рух (базовий)", "baseline_weapon": "Зброя (базовий)"}

    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))  # Ukrainian decimal comma on numeric axes

    def save(fig, name: str) -> None:
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                if isinstance(axis.get_major_formatter(), ScalarFormatter):
                    axis.set_major_formatter(comma)
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", facecolor="white")
        plt.close(fig)

    def roc(result: dict | None, pick: Callable[[dict], dict], title: str, name: str) -> None:
        if result is None:
            missing.append(name)
            return
        fig, ax = plt.subplots(figsize=(5, 5))
        for key in ("A", "baseline_motion", "baseline_weapon"):
            r = pick(result["configs"][key])
            if r.get("roc"):
                ax.plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{names[key]} (AUC {_n(r['auc'])})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
        ax.set(xlabel="Частка хибних спрацювань", ylabel="Повнота", title=title, xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(loc="lower right", fontsize=8)
        save(fig, name)

    roc(_json(EXPERIMENT / "rwf_test.json"), lambda r: r, "ROC: RWF-2000, тест (епізоди)", "roc_rwf_test")
    roc(
        _json(EXPERIMENT / "xd_test.json"), lambda r: r["window"]["all"], "ROC: XD-Violence (вікна)", "roc_xd"
    )

    sweep = _json(RESULTS / "weapon_crops_sweep_2026-10-03.json")
    base = _json(RESULTS / "baselines_2026-09-30.json")
    rule = _json(RESULTS / "weapon_rule_2026-09-30.json")
    if sweep and base and rule:
        fig, ax = plt.subplots(figsize=(6, 4))
        rows = sweep["rows"]
        ax.plot(
            [r["tau"] for r in rows],
            [r["recall"] for r in rows],
            marker="o",
            label="Вирізки 320:2, правило (2 вікна)",
        )
        grid = [g for g in base["weapon"]["grid"] if g["segment_recall"] is not None]
        ax.plot(
            [g["threshold"] for g in grid],
            [g["segment_recall"] for g in grid],
            marker=".",
            label="Кадр 480:2, без підтвердження",
        )
        r0 = rule["кадр 480:2"]["confirm2"]
        ax.scatter([0.30], [r0["recall"]], color="black", zorder=5, label="Кадр 480:2, правило (обрано)")
        ax.set(xlabel="Поріг детектора зброї", ylabel="Повнота сегментів (Севілья dev)", ylim=(0, 1.02))
        ax.legend(fontsize=8)
        save(fig, "weapon_recall_vs_threshold")
    else:
        missing.append("weapon_recall_vs_threshold")

    fsm = _json(RESULTS / "fsm_tuning_2026-09-30.json")
    if fsm:
        fig, ax = plt.subplots(figsize=(6, 4))
        for quiet in sorted({r["quiet_s"] for r in fsm["rows"]}):
            rows = sorted((r for r in fsm["rows"] if r["quiet_s"] == quiet), key=lambda r: r["theta_low"])
            ax.plot(
                [r["theta_low"] for r in rows],
                [r["handled"] for r in rows],
                marker="o",
                label=f"M = {quiet:g} с",
            )
        ax.set(xlabel="θ_low", ylabel="Частка опрацьованих бійок (склейки val)", ylim=(0, 1))
        ax.legend(fontsize=8)
        save(fig, "theta_low_tuning")
    else:
        missing.append("theta_low_tuning")

    path = RESULTS / "bench_quiet_2026-10-03.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = [
            f"{r['people'].replace(' detect', '').replace(' track', '')}\n{r['weapons']} {r['backend']}"
            for r in rows
        ]
        ax.barh(
            range(len(rows)),
            [r["rtf"] for r in rows],
            color=["#3b6ea8" if r["backend"] == "openvino" else "#a8693b" for r in rows],
        )
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(rows)), labels, fontsize=6)
        ax.set(
            xlabel="RTF (менше — швидше; 1 — реальний час)", title="Швидкість конфігурацій (CPU, 10 кадрів/с)"
        )
        save(fig, "speed_by_configuration")
    else:
        missing.append("speed_by_configuration")

    auc = _json(RESULTS / "features_auc_train_2026-09-30.json")
    if auc:
        rows = sorted((r for r in auc["windows"] if r["auc"] is not None), key=lambda r: r["auc"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([r["feature"] for r in rows], [r["auc"] for r in rows], color="#3b6ea8")
        ax.axvline(0.5, color="grey", linestyle=":")
        ax.set(xlabel="AUC вікон однієї ознаки (RWF train)", xlim=(0.3, 0.85))
        ax.tick_params(axis="y", labelsize=7)
        save(fig, "feature_aucs")
    else:
        missing.append("feature_aucs")

    model = _json(Path("weights/aggregator_a/a_v1_1_all.json"))
    if model:
        pairs = sorted(zip(model["features"], model["weights"], strict=True), key=lambda p: p[1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            color=["#a83b3b" if p[1] > 0 else "#3b6ea8" for p in pairs],
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Коефіцієнт (стандартизовані ознаки)", title="Агрегатор A v1.1")
        ax.tick_params(axis="y", labelsize=7)
        save(fig, "a_v1_1_coefficients")
    else:
        missing.append("a_v1_1_coefficients")

    grid = sorted(RESULTS.glob("cascade_grid_*.json"))
    if grid:
        rows = _json(grid[-1]).get("rows", [])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([r["b_share"] for r in rows], [r["auc"] for r in rows], s=12)
        ax.set(xlabel="Частка вікон, переданих B", ylabel="AUC вікон каскаду (val)")
        save(fig, "cascade_curve")
    else:
        missing.append("cascade_curve (чекає B на val і cascade-grid)")
    return missing


# --- screens and listings -------------------------------------------------------------------------------


def screens(folder: Path, demo: Path = RESULTS / "demo") -> list[str]:
    """Key frames (faces blurred) of the demo incidents with their message texts, and one overlay frame."""
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    notes = []
    for sidecar in sorted(demo.glob("*.events.json")):
        log = json.loads(sidecar.read_text(encoding="utf-8"))
        video = sidecar.with_name(sidecar.name.removesuffix(".events.json") + ".mp4")
        opened = [e for e in log["events"] if e["kind"] in ("open", "escalate")]
        for i, event in enumerate(opened[:1] + [e for e in log["events"] if e["kind"] == "close"][:1]):
            stem = f"{video.stem}_{event['kind']}"
            if event.get("keyframe") and Path(event["keyframe"]).exists():
                shutil.copy(event["keyframe"], folder / f"{stem}_keyframe.jpg")
            (folder / f"{stem}.txt").write_text(event["text"] + "\n", encoding="utf-8")
            notes.append(f"{stem}: {event['text'].splitlines()[0]}")
            if i == 0 and video.exists():  # the overlay 2 s after the alarm: banner, risk, the message card
                capture = cv2.VideoCapture(str(video))
                capture.set(cv2.CAP_PROP_POS_FRAMES, event["frame"] + 20)
                ok, frame = capture.read()
                capture.release()
                if ok:
                    cv2.imwrite(str(folder / f"{video.stem}_overlay.png"), frame)
    return notes


def listings(folder: Path) -> None:
    from sentinel import incidents, schema
    from sentinel.aggregators import rules
    from sentinel.aggregators.llm import PROMPTS

    folder.mkdir(parents=True, exist_ok=True)
    contract = [
        inspect.getsource(c)
        for c in (
            schema.Window,
            schema.Zone,
            schema.Summary,
            schema.Person,
            schema.Interaction,
            schema.SceneDescription,
            schema.RiskResult,
        )
    ]
    (folder / "scene_description.py").write_text("\n\n".join(contract), encoding="utf-8")
    (folder / "scene_to_features.py").write_text(inspect.getsource(rules.scene_to_features), encoding="utf-8")
    (folder / "incident_automaton.py").write_text(
        inspect.getsource(incidents.WeaponConfirmation)
        + "\n\n"
        + inspect.getsource(incidents.IncidentManager.on_window),
        encoding="utf-8",
    )
    shutil.copy(PROMPTS / "b_v1.md", folder / "prompt_b_v1.md")


def env_md(env: dict) -> str:
    freeze = _json(Path("docs/experiment_freeze.json"))
    image = subprocess.run(
        ["docker", "image", "inspect", "-f", "{{.Id}} {{.Created}}", "sentinel:dev"],
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    lines = [
        "# Середовище, хеші, відтворення",
        "",
        f"- Коміт: `{commit}`; пакет зібрано {datetime.now().strftime('%d.%m.%Y %H:%M')}.",
        f"- Docker-образ `sentinel:dev`: `{image.stdout.strip() or '—'}`.",
        f"- Python {env['python']}; " + ", ".join(f"{k} {v}" for k, v in env["versions"].items()) + ".",
        f"- Конфіг SHA-256 `{env['config_sha256']}`; A v1.1 `{env['a_sha256']}`.",
        "- Ваги: " + "; ".join(f"{k} `{v}`" for k, v in env["weights_sha256"].items()) + ".",
        "",
        "## Заморожене перед відкриттям тесту",
        "",
        "```json",
        json.dumps(freeze, ensure_ascii=False, indent=1) if freeze else "—",
        "```",
        "",
        "## Команди відтворення",
        "",
        "```bash",
        "uv sync                                          # залежності з uv.lock (torch — CPU-збірка)",
        "uv run sentinel export                           # експорт моделей в OpenVINO",
        "uv run sentinel eval extract …                   # кеш описів сцен (див. README)",
        "uv run sentinel eval train-a                     # A v1.1 на train, вибір на val",
        "uv run sentinel eval baselines                   # базові методи, пороги на val",
        "uv run sentinel eval fsm-tune                    # θ_low і M на склейках val",
        "uv run sentinel eval experiment rwf|sevilla|xd   # тест (перевіряє хеші заморожування)",
        "uv run sentinel eval experiment report",
        "scripts/protocol.sh --build                      # протокол випробувань у Docker",
        "uv run --with matplotlib sentinel eval thesis-pack",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=RESULTS / f"thesis_pack_{date.today().isoformat()}.zip")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / args.out.stem
        root.mkdir()
        numbers = Numbers()
        env = machine(numbers)
        speed_rows = speed(numbers)
        dataset_rows = datasets(numbers)
        quality(numbers)
        automaton(numbers, config)
        aggregator_b(numbers)
        code_size(numbers)
        tables(root / "tables", speed_rows, dataset_rows)
        missing = figures(root / "figures")
        shots = screens(root / "screens")
        listings(root / "listings")
        (root / "env.md").write_text(env_md(env), encoding="utf-8")
        copies = root / "docs"
        (copies / "reports").mkdir(parents=True)
        for path in [
            Path("docs/SPEC.md"),
            Path("docs/DECISIONS.md"),
            Path("docs/demo.md"),
            *sorted(Path("docs/bench").glob("experiment_test_*.md")),
        ]:
            if path.exists():
                shutil.copy(path, copies / path.name)
        for path in sorted(Path("docs/reports").glob("*.md")):
            shutil.copy(path, copies / "reports" / path.name)
        numbers.lines += [
            "",
            "## Що в пакеті",
            "",
            f"- Скриншоти: {len(shots)}" + "".join(f"\n  - {s}" for s in shots),
        ]
        numbers.lines += [
            f"- Рисунків немає: {', '.join(missing)}" if missing else "- Усі рисунки побудовано."
        ]
        (root / "numbers.md").write_text(
            numbers.text(f"Ключові числа Sentinel — {date.today().strftime('%d.%m.%Y')}"), encoding="utf-8"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(root.parent))
        args.out.write_bytes(buffer.getvalue())
    print(f"{args.out}: {len(buffer.getvalue()) / 1e6:.1f} МБ; рисунків бракує: {missing or 'немає'}")


if __name__ == "__main__":
    main()
