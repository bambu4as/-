"""
ablation_ind.py — обучение на наборе, который будет в работе: доходности и основные индикаторы (ind_main8).

  python3 ml/ablation_ind.py --csv-dir /path/to/otc --out train/out/otc/ablation_ind

Варианты:
  R    ret20                       базовый (то, что было)
  I8   ret20 + ind_main8           все 8 индикаторов
  I5   ret20 + 5 основных          rsi7, rsi14, stoch14, macd_hist, bb_z20
  Ionly ind_main8                  только индикаторы
Оценка: все сделки валидации (CALL при P>=0.5), тест не используется.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

from ablation import fit_predict, summarize  # noqa: E402
from indicators import IND_FEATURES  # noqa: E402
from prepare import build_pair_ind, load_ohlc, pair_name  # noqa: E402

SUB5 = [IND_FEATURES.index(k) for k in ("rsi7", "rsi14", "stoch14", "macd_hist", "bb_z20")]
VARIANTS = {
    "R ret20":        lambda X, I: X,
    "I8 ret20+ind8":  lambda X, I: np.hstack([X, I]),
    "I5 ret20+ind5":  lambda X, I: np.hstack([X, I[:, SUB5]]),
    "Ionly ind8":     lambda X, I: I,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-pair", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    tr_X, tr_I, tr_y = [], [], []
    va_X, va_I, va_y, va_p = [], [], [], []
    for f in sorted(glob.glob(os.path.join(args.csv_dir, "hist_*_1m.csv"))):
        name = pair_name(f)
        t, o, h, l, c = load_ohlc(f)
        X, I, y, tt, s = build_pair_ind(t, o, h, l, c)
        tri = rng.choice(np.where(s == 0)[0], min(args.train_per_pair, int((s == 0).sum())), replace=False)
        vai = np.where(s == 1)[0]
        tr_X.append(X[tri]); tr_I.append(I[tri]); tr_y.append(y[tri])
        va_X.append(X[vai]); va_I.append(I[vai]); va_y.append(y[vai])
        va_p.append(np.full(len(vai), name, dtype=object))
        print(f"{name}: train {len(tri)}, val {len(vai)}", flush=True)

    Xtr = np.vstack(tr_X); Itr = np.vstack(tr_I); ytr = np.concatenate(tr_y)
    Xva = np.vstack(va_X); Iva = np.vstack(va_I); yva = np.concatenate(va_y); pva = np.concatenate(va_p)
    print(f"всего train {len(ytr)}, val {len(yva)}\n", flush=True)

    results = []
    for vname, build in VARIANTS.items():
        A = build(Xtr, Itr); B = build(Xva, Iva)
        for mname in ("logistic", "rf"):
            ptr, pv = fit_predict(mname, A, ytr, B, seed=args.seed)
            r_val = summarize(pv, yva)
            r_tr = summarize(ptr, ytr)
            per_pair = {str(nm): summarize(pv[pva == nm], yva[pva == nm])["acc"] for nm in np.unique(pva)}
            order = np.argsort(pv)
            deciles = [float(((pv[ch] >= 0.5).astype(int) == yva[ch]).mean()) for ch in np.array_split(order, 10)]
            results.append({"variant": vname, "model": mname, "n_features": A.shape[1],
                            "val": r_val, "train_acc": r_tr["acc"], "per_pair": per_pair, "deciles_acc": deciles})
            print(f"{vname:14s} {mname:9s} признаков {A.shape[1]:3d}  val {100*r_val['acc']:6.2f}%  "
                  f"95%: {100*r_val['ci_low']:5.2f}-{100*r_val['ci_high']:5.2f}  p={r_val['p']:.3g}  "
                  f"train {100*r_tr['acc']:6.2f}%", flush=True)

    with open(os.path.join(args.out, "ablation.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.join(args.out, 'ablation.json')}", flush=True)


if __name__ == "__main__":
    main()
