"""
arrays.py — подготовка признаков для деревьев и базовых линий напрямую из CSV (без промптов).

Правила те же, что в train/build_dataset.py:
  * пример: окно из 21 закрытия -> 20 доходностей (б.п.), метка — направление следующего бара;
  * сплит по времени 70/15/15, пример целиком в одном сплите (окно и метка);
  * ничьи (close == close) отбрасываются;
  * доходности округляются как Math.round в JavaScript (совпадает с buildRequest в llmCore.js).
"""
import csv
import math
import os

import numpy as np

from features import FEATURE_LEN

SPLIT_RATIO = (0.70, 0.15)


def read_closes(path):
    """Возвращает (times, closes, highs, lows) по возрастанию времени; дубли — последний побеждает."""
    bars = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = int(float(row["time"]))
            bars[t] = (float(row["close"]), float(row["high"]), float(row["low"]))
    times = np.array(sorted(bars), dtype=np.int64)
    closes = np.array([bars[int(t)][0] for t in times], dtype=np.float64)
    return times, closes


def js_round(x):
    """Math.round из JavaScript (x.5 -> +∞). np.round использует банковское округление — не подходит."""
    return np.floor(x + 0.5)


def returns_bp(closes):
    """Доходности в б.п., по одной на пару соседних закрытий. Та же операция, что в JavaScript."""
    return js_round((closes[1:] / closes[:-1] - 1.0) * 1e4)


def build(times, closes, step):
    """
    Возвращает dict: X (N x 20 float32), y (0/1), split (0 train, 1 val, 2 test), t (время метки).
    Окно i-20..i, метка по бару i+1. Пропуски и ничьи отбрасываются.
    """
    n = len(closes)
    b1 = int(n * SPLIT_RATIO[0])
    b2 = int(n * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))
    r = returns_bp(closes)                    # r[k] = доходность между k и k+1
    gaps = np.diff(times) != step             # разрыв между k и k+1
    gap_prefix = np.concatenate([[0], np.cumsum(gaps)])

    X, y, split, t_out = [], [], [], []
    for i in range(FEATURE_LEN, n - 1):
        start = i - FEATURE_LEN
        if gap_prefix[i] - gap_prefix[start] != 0 or gaps[i]:
            continue                          # разрыв внутри окна или перед меткой
        if closes[i + 1] == closes[i]:
            continue                          # ничья
        if i + 1 < b1:
            s = 0
        elif start >= b1 and i + 1 < b2:
            s = 1
        elif start >= b2:
            s = 2
        else:
            continue                          # граница сплитов, пример отбрасываем
        X.append(r[start:i])
        y.append(1 if closes[i + 1] > closes[i] else 0)
        split.append(s)
        t_out.append(times[i])
    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int8),
        "split": np.asarray(split, dtype=np.int8),
        "t": np.asarray(t_out, dtype=np.int64),
    }


def save(path, arrays, pair):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, pair=np.array(pair), **arrays)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in ("X", "y", "split", "t")}, str(z["pair"])
