"""
prepare.py — числовой датасет из CSV истории (без JSON-промптов).

  python3 ml/prepare.py --csv-dir /path/to/otc --segment otc --pairs EUR_USD,GBP_USD

Для каждой пары:
  * признаки: 20 доходностей в б.п., формула как в buildRequest() (JS Math.round);
  * метка: направление следующего бара (ничьи отбрасываются);
  * разбиение по времени внутри пары: train 70% / val 15% / test 15%, пересекающие границы примеры отбрасываются.
Итог: <out>/<ПАРА>.npz по парам и <out>/merged.npz со всеми парами (колонка pair — индекс пары).
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import FEATURE_LEN, FEATURE_VERSION  # noqa: E402

TRAIN, VAL, TEST = 0, 1, 2


def pair_name(path):
    """hist_EUR_USD_OTC_1m.csv -> 'EUR/USD OTC'."""
    base = os.path.basename(path)
    core = base[len("hist_"):].rsplit("_", 1)[0]          # EUR_USD_OTC
    parts = core.split("_")
    return f"{parts[0]}/{parts[1]} {parts[2]}" if len(parts) == 3 else core


def load_closes(path):
    times, closes = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            times.append(int(float(row["time"])))
            closes.append(float(row["close"]))
    order = np.argsort(times, kind="stable")
    t = np.asarray(times, dtype=np.int64)[order]
    c = np.asarray(closes, dtype=np.float64)[order]
    keep = np.concatenate([[True], np.diff(t) > 0])       # дубли по времени убираем
    return t[keep], c[keep]


def load_ohlc(path):
    """Время, open, high, low, close по возрастанию времени, дубли убраны."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(float(row["time"])), float(row["open"]), float(row["high"]),
                         float(row["low"]), float(row["close"])))
    rows.sort(key=lambda r: r[0])
    arr = np.asarray(rows, dtype=np.float64)
    keep = np.concatenate([[True], np.diff(arr[:, 0]) > 0])
    arr = arr[keep]
    return (arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])


EXTRA_NAMES = ["rng_last", "rng_mean20", "atr14", "tod_sin", "tod_cos"]


def extra_features(t, o, h, l, c, i):
    """
    Дополнительные признаки для решения на баре i. Используются только бары <= i (без заглядывания вперёд).
    Диапазон и ATR в б.п. от close(i). Для этапа абляции; в JS пока не реализованы.
    Возвращает: rng20 (20 значений диапазона окна), rng_last, rng_mean20, atr14, tod_sin, tod_cos.
    """
    start = i - FEATURE_LEN
    rng = (h[start:i + 1] - l[start:i + 1]) / c[start:i + 1] * 1e4          # 21 значение, бары start..i
    rng20 = rng[1:]                                                          # 20 значений окна
    k0 = i - 13
    tr = []
    for k in range(k0, i + 1):
        pc = c[k - 1]
        tr.append(max(h[k] - l[k], abs(h[k] - pc), abs(l[k] - pc)))
    atr14 = float(np.mean(tr)) / c[i] * 1e4
    minute_of_day = (int(t[i]) // 60) % 1440
    ang = 2 * np.pi * minute_of_day / 1440.0
    return rng20, float(rng20[-1]), float(np.mean(rng20)), atr14, float(np.sin(ang)), float(np.cos(ang))


def returns_bp(closes):
    """Совпадает с JS: Math.round((c[i]/c[i-1]-1)*1e4), .5 -> +inf."""
    r = (closes[1:] / closes[:-1] - 1.0) * 1e4
    return np.floor(r + 0.5)


def build_pair(times, closes, step=60, ratio=(0.70, 0.15)):
    n = len(closes)
    rb = returns_bp(closes)                                # rb[j-1] = bp между j-1 и j
    b1 = int(n * ratio[0])
    b2 = int(n * (ratio[0] + ratio[1]))
    X, y, t, s = [], [], [], []
    dropped = {"gap": 0, "tie": 0, "purge": 0}
    for i in range(FEATURE_LEN, n - 1):
        start = i - FEATURE_LEN
        if times[i + 1] - times[i] != step or np.any(np.diff(times[start:i + 2]) != step):
            dropped["gap"] += 1
            continue
        if closes[i + 1] == closes[i]:
            dropped["tie"] += 1
            continue
        if i + 1 < b1:
            split = TRAIN
        elif start >= b1 and i + 1 < b2:
            split = VAL
        elif start >= b2:
            split = TEST
        else:
            dropped["purge"] += 1
            continue
        # доходности окна: rb[start .. i-1] дают FEATURE_LEN значений (закрытия start..i)
        X.append(rb[start:i])
        y.append(1 if closes[i + 1] > closes[i] else 0)
        t.append(times[i])
        s.append(split)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8),
            np.asarray(t, dtype=np.int64), np.asarray(s, dtype=np.int8), dropped)


def build_pair_ind(t, o, h, l, c, step=60, ratio=(0.70, 0.15)):
    """
    Как build_pair, плюс основной набор индикаторов ind_main8 на баре решения (см. ml/indicators.py).
    Возвращает X (20 доходностей), I (8 индикаторов), y, t, split.
    """
    from indicators import IND_FEATURES, main_indicators
    X, y, tt, s, _ = build_pair(t, c, step=step, ratio=ratio)
    ind = main_indicators(h, l, c)
    index = {int(v): k for k, v in enumerate(t)}
    rows = np.array([index[int(ti)] for ti in tt], dtype=np.int64)
    I = np.column_stack([ind[k][rows] for k in IND_FEATURES]).astype(np.float64)
    # Прогрев: MACD нужен 35 баров. В работе при меньшем числе баров решения нет, поэтому пример исключается.
    ok = ~np.isnan(I).any(axis=1)
    return X[ok], I[ok].astype(np.float32), y[ok], tt[ok], s[ok]


def build_pair_ext(t, o, h, l, c, step=60, ratio=(0.70, 0.15)):
    """
    Как build_pair, но дополнительно отдаёт расширенные признаки.
    Возвращает X (20 доходностей), E (rng20 + 5 скаляров = 25), y, t, split.
    Разбиение, ничьи и разрывы — те же, что в build_pair.
    """
    X, y, tt, s, _ = build_pair(t, c, step=step, ratio=ratio)
    index = {int(v): k for k, v in enumerate(t)}
    E = []
    for ti in tt:
        i = index[int(ti)]
        rng20, rl, rm, atr, sn, cs = extra_features(t, o, h, l, c, i)
        E.append(np.concatenate([rng20, [rl, rm, atr, sn, cs]]))
    return X, np.asarray(E, dtype=np.float32), y, tt, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--segment", required=True, choices=["currency", "otc"])
    ap.add_argument("--pairs", help="фильтр по подстроке, через запятую (EUR_USD,GBP_USD); по умолчанию все *_1m.csv")
    ap.add_argument("--out", help="по умолчанию train/data/<сегмент>/arrays")
    args = ap.parse_args()

    out = args.out or os.path.join(ROOT, "train", "data", args.segment, "arrays")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.csv_dir, "hist_*_1m.csv")))
    if args.pairs:
        keys = [k.strip() for k in args.pairs.split(",") if k.strip()]
        files = [f for f in files if any(k in os.path.basename(f) for k in keys)]
    if not files:
        raise SystemExit("нет подходящих файлов *_1m.csv")

    parts, names = [], []
    for f in files:
        name = pair_name(f)
        times, closes = load_closes(f)
        t_, o_, h_, l_, c_ = load_ohlc(f)
        X, I, y, t, s = build_pair_ind(t_, o_, h_, l_, c_)
        dropped = {}
        np.savez(os.path.join(out, name.replace("/", "_").replace(" ", "_") + ".npz"), X=X, I=I, y=y, t=t, split=s)
        print(f"{name:14s} примеров {len(y):7d}  train {int((s==TRAIN).sum()):7d}  "
              f"val {int((s==VAL).sum()):6d}  test {int((s==TEST).sum()):6d}  "
              f"доля CALL(train) {float(y[s==TRAIN].mean()):.3f}  отброшено {dropped}")
        parts.append((X, I, y, t, s, np.full(len(y), len(names), dtype=np.int16)))
        names.append(name)

    X = np.concatenate([p[0] for p in parts]); I = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts]); t = np.concatenate([p[3] for p in parts])
    s = np.concatenate([p[4] for p in parts]); pid = np.concatenate([p[5] for p in parts])
    np.savez(os.path.join(out, "merged.npz"), X=X, I=I, y=y, t=t, split=s, pair=pid)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"segment": args.segment, "pairs": names, "feature_version": FEATURE_VERSION,
                   "indicator_set": "ind_main8", "arrays": ["X (20 доходностей)", "I (8 индикаторов ind_main8)"],
                   "n_total": int(len(y)), "n_train": int((s == TRAIN).sum()),
                   "n_val": int((s == VAL).sum()), "n_test": int((s == TEST).sum())}, fh, ensure_ascii=False, indent=2)
    print(f"\nвсего: {len(y)} примеров, пар {len(names)} -> {os.path.join(out, 'merged.npz')}")


if __name__ == "__main__":
    main()
