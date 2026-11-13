"""
baselines.py — простые правила и логистическая регрессия на числовых массивах (prepare.py).

  python3 ml/baselines.py --arrays /path/merged.npz --split val    # разработка
  python3 ml/baselines.py --arrays /path/merged.npz --split test   # один раз, финальный отчёт

Показывает точность, 95% интервал Уилсона и p-value против 50%, и безубыточность при выплате.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))

from common import binom_two_sided_p, breakeven, wilson  # noqa: E402

PAYOUT_DEFAULT = 0.80


def report(name, pred, y, payout):
    """pred: массив 1 (CALL), 0 (PUT), -1 (NONE)."""
    trade = pred >= 0
    n = int(trade.sum())
    k = int(((pred == y) & trade).sum())
    lo, hi = wilson(k, n)
    acc = k / n if n else float("nan")
    return (f"{name:28s} сделок {n:7d} из {len(y)}  точность {100 * acc:6.2f}%  "
            f"95%: {100 * lo:5.2f}-{100 * hi:5.2f}  p={binom_two_sided_p(k, n):.3g}  "
            f"безуб. {100 * breakeven(payout):.2f}%"), acc, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--payout", type=float, default=PAYOUT_DEFAULT)
    args = ap.parse_args()

    d = np.load(args.arrays)
    sid = {"train": 0, "val": 1, "test": 2}
    tr = d["split"] == 0
    ev = d["split"] == sid[args.split]
    Xtr, ytr = d["X"][tr].astype(np.float64), d["y"][tr].astype(int)
    Xev, yev = d["X"][ev].astype(np.float64), d["y"][ev].astype(int)
    print(f"сплит {args.split}: {len(yev)} примеров, доля CALL {yev.mean():.3f}")

    rows = []
    rows.append(report("always CALL", np.ones_like(yev), yev, args.payout))
    rows.append(report("always PUT", np.zeros_like(yev), yev, args.payout))
    last = Xev[:, -1]
    mom = np.where(last > 0, 1, np.where(last < 0, 0, -1))
    rev = np.where(last > 0, 0, np.where(last < 0, 1, -1))
    rows.append(report("momentum (знак последней)", mom, yev, args.payout))
    rows.append(report("reversal (против последней)", rev, yev, args.payout))

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500).fit(Xtr / 100.0, ytr)
    lr = clf.predict(Xev / 100.0)
    rows.append(report("logistic (20 доходностей)", lr, yev, args.payout))

    for line, _, _ in rows:
        print("  " + line)
    best = max(acc for _, acc, n in rows if n > 0)
    print(f"\nлучшая базовая линия: {100 * best:.2f}%  (безубыточность {100 * breakeven(args.payout):.2f}%)")


if __name__ == "__main__":
    main()
