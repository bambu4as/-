"""
ablation.py — абляция признаков по всем парам сегмента, точность на ВСЕХ сделках валидации.

  python3 ml/ablation.py --csv-dir /path/to/otc --out train/out/otc/ablation

Варианты признаков (тест не используется):
  A  ret20                     20 доходностей (базовый)
  B  ret20 + rng20             + диапазон каждого бара окна
  C  ret20 + rng_last/mean/atr + диапазон последнего бара, среднее, ATR14
  D  ret20 + время суток       sin/cos минуты UTC
  E  ret20 + rng20 + C + D     всё вместе
  F  ret20 + atr14             только ATR

Модели: логистическая регрессия (стандартизованные признаки) и Random Forest.
Обучение: случайная подвыборка train по парам (одинаковая для всех вариантов).
Оценка: каждая сделка валидации (решение CALL при P>=0.5, без воздержания).
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

from common import binom_two_sided_p, wilson  # noqa: E402
from prepare import build_pair_ext, load_ohlc, pair_name  # noqa: E402

VARIANTS = {
    "A ret20":            lambda X, E: X,
    "B ret20+rng20":      lambda X, E: np.hstack([X, E[:, 0:20]]),
    "C ret20+rng+atr":    lambda X, E: np.hstack([X, E[:, 20:23]]),
    "D ret20+tod":        lambda X, E: np.hstack([X, E[:, 23:25]]),
    "E all":              lambda X, E: np.hstack([X, E]),
    "F ret20+atr":        lambda X, E: np.hstack([X, E[:, 22:23]]),
}


def fit_predict(model_name, Xtr, ytr, Xva, seed=0):
    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    else:
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=50,
                                   random_state=seed, n_jobs=1)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xva)[:, 1]


def summarize(p, y):
    pred = (p >= 0.5).astype(int)
    n = len(y)
    k = int((pred == y).sum())
    lo, hi = wilson(k, n)
    return {"n": n, "acc": k / n, "ci_low": lo, "ci_high": hi, "p": binom_two_sided_p(k, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-pair", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    tr_parts, va_parts = [], []
    for f in sorted(glob.glob(os.path.join(args.csv_dir, "hist_*_1m.csv"))):
        name = pair_name(f)
        t, o, h, l, c = load_ohlc(f)
        X, E, y, tt, s = build_pair_ext(t, o, h, l, c)
        tri = np.where(s == 0)[0]
        tri = rng.choice(tri, min(args.train_per_pair, len(tri)), replace=False)
        vai = np.where(s == 1)[0]
        tr_parts.append((X[tri], E[tri], y[tri]))
        va_parts.append((X[vai], E[vai], y[vai], np.full(len(vai), name, dtype=object)))
        print(f"{name}: train подвыборка {len(tri)}, val {len(vai)}", flush=True)

    Xtr_ret = np.vstack([p[0] for p in tr_parts]); Etr = np.vstack([p[1] for p in tr_parts])
    ytr = np.concatenate([p[2] for p in tr_parts])
    Xva_ret = np.vstack([p[0] for p in va_parts]); Eva = np.vstack([p[1] for p in va_parts])
    yva = np.concatenate([p[2] for p in va_parts]); pva = np.concatenate([p[3] for p in va_parts])
    print(f"\nвсего train {len(ytr)}, val {len(yva)}, пар {len(tr_parts)}", flush=True)

    results = []
    for vname, build in VARIANTS.items():
        Xtr = build(Xtr_ret, Etr); Xva = build(Xva_ret, Eva)
        for mname in ("logistic", "rf"):
            ptr, pva_p = fit_predict(mname, Xtr, ytr, Xva, seed=args.seed)
            r_val = summarize(pva_p, yva)
            r_tr = summarize(ptr, ytr)
            per_pair = {}
            for nm in np.unique(pva):
                m = pva == nm
                per_pair[str(nm)] = summarize(pva_p[m], yva[m])["acc"]
            # калибровка: точность по децилям P(CALL)
            order = np.argsort(pva_p)
            deciles = []
            for chunk in np.array_split(order, 10):
                deciles.append(float((((pva_p[chunk] >= 0.5).astype(int)) == yva[chunk]).mean()))
            results.append({"variant": vname, "model": mname, "n_features": Xtr.shape[1],
                            "val": r_val, "train_acc": r_tr["acc"], "per_pair": per_pair,
                            "deciles_acc": deciles})
            print(f"{vname:18s} {mname:9s} val {100*r_val['acc']:6.2f}%  "
                  f"95%: {100*r_val['ci_low']:5.2f}-{100*r_val['ci_high']:5.2f}  p={r_val['p']:.3g}  "
                  f"train {100*r_tr['acc']:6.2f}%", flush=True)

    with open(os.path.join(args.out, "ablation.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {os.path.join(args.out, 'ablation.json')}")


if __name__ == "__main__":
    main()
