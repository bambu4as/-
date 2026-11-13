"""
train_arrays.py — обучение на числовых массивах (prepare.py) для выбранного набора признаков.

  python3 ml/train_arrays.py --arrays train/data/otc/arrays/merged.npz --feature-set returns20+ind_main8 --models rf

Обучение: train (подвыборка --train-max), выбор ничего не делает: решение на КАЖДОЙ сделке (порог 0, ничья CALL).
Оценка на val выводится. Тест не используется (его оценка — отдельным шагом с одобрением).
Результат: <out>/bundle.joblib с полем feature_set (сервис проверяет совпадение).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from features import FEATURE_SETS, FEATURE_VERSION, IND_FEATURES  # noqa: E402
from models import make_models, predict_p_call  # noqa: E402


def matrix(d, feature_set, idx=None):
    X = d["X"] if idx is None else d["X"][idx]
    if feature_set == "returns20":
        return X.astype(np.float64)
    I = d["I"] if idx is None else d["I"][idx]
    return np.hstack([X, I]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--feature-set", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--models", default="rf")
    ap.add_argument("--train-max", type=int, default=300000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.arrays)
    if args.feature_set != "returns20" and "I" not in d.files:
        raise SystemExit("в массивах нет индикаторов: перезапустите ml/prepare.py")
    rng = np.random.RandomState(args.seed)
    tr = np.where(d["split"] == 0)[0]
    va = np.where(d["split"] == 1)[0]
    tr = rng.choice(tr, min(args.train_max, len(tr)), replace=False)

    Xtr = matrix(d, args.feature_set, tr); ytr = d["y"][tr].astype(int)
    Xva = matrix(d, args.feature_set, va); yva = d["y"][va].astype(int)
    print(f"набор {args.feature_set}: признаков {Xtr.shape[1]}, train {len(ytr)}, val {len(yva)}", flush=True)

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    models = make_models(names, seed=args.seed)
    for name, m in models.items():
        print(f"обучение {name} ...", flush=True)
        m.fit(Xtr, ytr)

    p = predict_p_call(models, Xva)
    pred = (p >= 0.5).astype(int)
    acc_val = float((pred == yva).mean())
    ptr = predict_p_call(models, Xtr)
    acc_tr = float(((ptr >= 0.5).astype(int) == ytr).mean())
    print(f"val (все сделки): точность {100*acc_val:.2f}%   train: {100*acc_tr:.2f}%", flush=True)

    os.makedirs(args.out, exist_ok=True)
    bundle = {"segment": None, "models": models, "threshold": 0.0, "mode": "all_trades",
              "feature_set": args.feature_set, "feature_version": FEATURE_VERSION,
              "indicator_names": IND_FEATURES if args.feature_set != "returns20" else [],
              "model_names": names, "trained_at": datetime.now(timezone.utc).isoformat(),
              "n_train": int(len(ytr)), "n_val": int(len(yva))}
    joblib.dump(bundle, os.path.join(args.out, "bundle.joblib"))
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in bundle.items() if k != "models"} | {"val_acc_all_trades": acc_val, "train_acc": acc_tr},
                  fh, ensure_ascii=False, indent=2)
    print(f"бандл: {os.path.join(args.out, 'bundle.joblib')}", flush=True)


if __name__ == "__main__":
    main()
