"""
import_history.py — приведение файлов истории к единому виду для обучения.

Понимает:
  * CSV: колонки time,open,high,low,close (остальные игнорируются);
  * JSON: массив свечей или объект с полем candles/data/history; поля time|t|timestamp, open, high, low, close.
Время в секундах; если значение больше 1e12, считается миллисекундами и делится на 1000.

  python3 tools/import_history.py data/history/*.json --pair "EUR/USD OTC" --period 60
  python3 tools/import_history.py old.csv --pair "GBP/USD OTC" --period 60

Результат: data/<сегмент>/hist_<ASSET>_<период>.csv (по возрастанию времени, без дублей).
Файл, в котором шаг времени не совпадает с периодом, в обучение не попадает.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
from common import segment_of  # noqa: E402

FIELDS = ("time", "open", "high", "low", "close")


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("candles", "data", "history"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise SystemExit(f"{path}: ожидался массив свечей")
    return data


def normalize(rows):
    bars = {}
    skipped = 0
    for r in rows:
        t = r.get("time", r.get("t", r.get("timestamp")))
        try:
            t = float(t)
            o, h, l, c = (float(r[k]) for k in ("open", "high", "low", "close"))
        except (TypeError, ValueError, KeyError):
            skipped += 1
            continue
        if t > 1e12:                      # миллисекунды
            t /= 1000.0
        t = int(t)
        bars[t] = (o, h, l, c)            # дубликаты: последний побеждает
    return bars, skipped


def load_file(path):
    if path.lower().endswith(".csv"):
        return read_csv(path)
    if path.lower().endswith(".json"):
        return read_json(path)
    raise SystemExit(f"{path}: поддерживаются .csv и .json")


def asset_name(pair):
    return re.sub(r"[^A-Za-z0-9]+", "_", pair).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--period", type=int, default=60, help="период, сек")
    ap.add_argument("--out", help="по умолчанию data/<сегмент>")
    args = ap.parse_args()

    segment = segment_of(args.pair)
    out_dir = args.out or os.path.join(ROOT, "data", segment)
    os.makedirs(out_dir, exist_ok=True)

    bars = {}
    skipped = 0
    for path in args.files:
        b, s = normalize(load_file(path))
        bars.update(b)                    # несколько файлов одной пары склеиваются
        skipped += s
        print(f"{path}: {len(b)} свечей, пропущено {s}")

    times = sorted(bars)
    if len(times) < 2:
        raise SystemExit("мало свечей")
    steps = Counter(b - a for a, b in zip(times, times[1:]))
    step, count = steps.most_common(1)[0]
    gaps = sum(v for k, v in steps.items() if k != args.period)
    share = gaps / max(1, sum(steps.values()))
    days = (times[-1] - times[0]) / 86400

    print(f"итого: {len(times)} свечей, {days:.1f} дн., шаг {step} с (ожидался {args.period} с)")
    print(f"разрывов/нестандартных шагов: {gaps} ({100 * share:.2f}%), пропущено строк: {skipped}")
    if step != args.period:
        raise SystemExit(f"самый частый шаг {step} с не совпадает с --period {args.period}: файлы не того периода")

    label = f"{args.period // 60}m" if args.period % 60 == 0 else f"{args.period}s"
    out = os.path.join(out_dir, f"hist_{asset_name(args.pair)}_{label}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close"])
        for t in times:
            o, h, l, c = bars[t]
            w.writerow([t, o, h, l, c])
    print(f"записано: {out}")
    if share > 0.02:
        print("ВНИМАНИЕ: больше 2% разрывов. Обучение на этих данных будет смещено.")


if __name__ == "__main__":
    main()
