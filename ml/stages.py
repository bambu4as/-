"""
stages.py — обучение ядра по этапам. Каждый этап решает свой вопрос и не лезет в тест раньше времени.

  prepare  --csv F --pair "EUR/USD OTC"             CSV -> data/arrays/<сегмент>/<пара>.npz
  stage1   --pair "EUR/USD OTC"                     есть ли сигнал на ОДНОЙ паре (только val)
  stage2   --segment otc --mode always|abstain      обучение деревьев на всех парах сегмента (threshold по val)
  stage3   --segment otc --approve                  ЕДИНСТВЕННАЯ оценка на тесте + одобрение

Режимы stage2:
  always   — ответ всегда CALL или PUT (порог 0), как просили. Допуск не смягчается.
  abstain  — порог воздержания подбирается на val.
Критерий допуска тот же (train/evaluate.py:gate): нижняя граница 95% > безубыточности, выше базовых линий.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

import arrays  # noqa: E402
from common import CONFIG, binom_two_sided_p, breakeven, wilson  # noqa: E402
from evaluate import gate  # noqa: E402
from features import FEATURE_VERSION  # noqa: E402

NONE, PUT, CALL = -1, 0, 1
DATA_ROOT = os.path.join(ROOT, "data", "arrays")


def pair_path(segment, pair):
    safe = pair.replace("/", "_").replace(" ", "_")
    return os.path.join(DATA_ROOT, segment, f"{safe}.npz")


def summary(name, pred, truth, payout):
    """Метрики в том же формате, что train/evaluate.summarize, но по массивам."""
    trade = pred != NONE
    n = int(trade.sum())
    k = int((pred[trade] == truth[trade]).sum()) if n else 0
    lo, hi = wilson(k, n)
    return {"name": name, "n_all": int(len(pred)), "n_trade": n,
            "coverage": n / len(pred) if len(pred) else float("nan"),
            "acc": k / n if n else float("nan"), "ci95": (lo, hi),
            "p_vs_50": binom_two_sided_p(k, n), "breakeven": breakeven(payout)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(m):
    if m["n_trade"] == 0:
        return f"{m['name']:30s} сделок 0"
    lo, hi = m["ci95"]
    return (f"{m['name']:30s} сделок {m['n_trade']:8d} (покрытие {100 * m['coverage']:5.1f}%) "
            f"точность {100 * m['acc']:6.2f}%  95%: {100 * lo:5.2f}-{100 * hi:5.2f}  p={m['p_vs_50']:.3g}")


def baselines(X_train, y_train, X, truth):
    """Базовые линии на наборе X. Возвращает dict name -> pred."""
    from sklearn.linear_model import LogisticRegression
    out = {}
    out["always CALL"] = np.full(len(truth), CALL)
    out["always PUT"] = np.full(len(truth), PUT)
    last = X[:, -1]
    mom = np.where(last > 0, CALL, np.where(last < 0, PUT, NONE))
    out["momentum (знак последней)"] = mom
    out["reversal (против последней)"] = np.where(mom == CALL, PUT, np.where(mom == PUT, CALL, NONE))
    clf = LogisticRegression(max_iter=1000).fit(X_train / 100.0, y_train)
    out["logistic на 20 доходностях"] = np.where(clf.predict(X / 100.0) == 1, CALL, PUT)
    return out


def cmd_prepare(args):
    times, closes = arrays.read_closes(args.csv)
    step = int(np.median(np.diff(times)))
    arr = arrays.build(times, closes, step)
    path = pair_path(args.segment, args.pair)
    arrays.save(path, arr, args.pair)
    counts = {n: int((arr["split"] == c).sum()) for c, n in ((0, "train"), (1, "val"), (2, "test"))}
    print(f"{args.pair}: шаг {step} с, примеров {counts} -> {path}")


def cmd_stage1(args):
    path = pair_path(args.segment, args.pair)
    if not os.path.exists(path):
        raise SystemExit(f"нет массивов: {path}. Сначала prepare")
    a, pair = arrays.load(path)
    tr, va = a["split"] == 0, a["split"] == 1
    X_tr, y_tr = a["X"][tr], a["y"][tr]
    X_va, y_va = a["X"][va], a["y"][va]
    truth = np.where(y_va == 1, CALL, PUT)
    print(f"стадия 1: {pair}, train {len(y_tr)}, val {len(y_va)}, доля CALL на val {y_va.mean():.3f}")
    print(f"безубыточность при выплате {CONFIG['payout']:.0%}: {100 * breakeven():.2f}%  (оценка только на val)\n")
    results = []
    for name, pred in baselines(X_tr, y_tr, X_va, truth).items():
        m = summary(name, pred, truth, CONFIG["payout"])
        results.append(m)
        print("  " + fmt(m))
    best = max((m for m in results if m["n_trade"] > 0), key=lambda m: m["acc"])
    lo = best["ci95"][0]
    print("\nвывод:")
    if lo > breakeven():
        print(f"  сигнал есть: лучшая линия '{best['name']}' выше безубыточности с запасом.")
    else:
        print(f"  сигнала, устойчиво выше безубыточности, не видно: нижняя граница лучшей линии "
              f"({100 * lo:.2f}%) не выше {100 * breakeven():.2f}%.")
        print("  Деревья могут его не найти. Решение о запуске стадии 2 принимайте по этой строке.")


def load_segment(segment, pairs):
    parts = {0: ([], []), 1: ([], []), 2: ([], [])}
    for pair in pairs:
        a, _ = arrays.load(pair_path(segment, pair))
        for code in parts:
            m = a["split"] == code
            parts[code][0].append(a["X"][m])
            parts[code][1].append(a["y"][m])
    return {c: (np.concatenate(x), np.concatenate(y)) for c, (x, y) in parts.items()}


def cmd_stage2(args):
    pairs = [p.strip() for p in args.pairs.split(",")] if args.pairs else prepared_pairs(args.segment)
    data = load_segment(args.segment, pairs)
    (Xtr, ytr), (Xva, yva) = data[0], data[1]
    if args.max_train_rows and len(ytr) > args.max_train_rows:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(ytr), args.max_train_rows, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"стадия 2: сегмент {args.segment}, пар {len(pairs)}, train {len(ytr)}, val {len(yva)}, режим {args.mode}")
    models = fit_models(args.models, Xtr, ytr, rf_trees=args.rf_trees, seed=args.seed)
    from models import predict_p_call  # noqa: E402
    p_va = predict_p_call(models, Xva)
    if args.mode == "always":
        thr, thr_info = 0.0, {"threshold": 0.0}
    else:
        thr, thr_info = pick_threshold_arrays(p_va, yva, args.min_val_trades, args.min_coverage)
    pred = decide_arrays(p_va, thr, args.mode)
    truth = np.where(yva == 1, CALL, PUT)
    m = summary(f"деревья ({args.mode}) на val", pred, truth, CONFIG["payout"])
    print("  " + fmt(m))
    out = os.path.join(ROOT, "train", "out", args.segment, "trees")
    os.makedirs(out, exist_ok=True)
    bundle = {"segment": args.segment, "models": models, "threshold": float(thr),
              "mode": args.mode, "feature_version": FEATURE_VERSION, "model_names": args.models.split(","),
              "pairs": pairs, "trained_at": datetime.now(timezone.utc).isoformat(),
              "n_train": int(len(ytr)), "n_val": int(len(yva)), "val": thr_info}
    path = os.path.join(out, "bundle.joblib")
    joblib.dump(bundle, path)
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "models"}, f, ensure_ascii=False, indent=2, default=str)
    print(f"бандл: {path}, порог {thr:.2f}, режим {args.mode}")
    print("дальше: stage3 (тест, один раз)")


def prepared_pairs(segment):
    """Все пары, для которых подготовлены массивы в сегменте."""
    d = os.path.join(DATA_ROOT, segment)
    if not os.path.isdir(d):
        raise SystemExit(f"нет подготовленных данных для сегмента {segment}: сначала prepare")
    names = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
    return [str(arrays.load(os.path.join(d, n))[1]) for n in names]


def decide_arrays(p, thr, mode):
    """
    Векторная копия features.decide: тот же ответ для каждого примера.
    Ничья (p == 0.5) -> CALL в режиме always, NONE в abstain.
    """
    edge = p - 0.5
    tie = CALL if mode == "always" else NONE
    call = (edge > 0) & (np.abs(edge) >= thr)
    put = (edge < 0) & (np.abs(edge) >= thr)
    return np.where(edge == 0, tie, np.where(call, CALL, np.where(put, PUT, NONE)))


def pick_threshold_arrays(p_va, yva, min_trades, min_cov):
    best, best_info = 0.0, {"threshold": 0.0}
    n = len(yva)
    for t in [round(x * 0.01, 2) for x in range(21)]:
        act = np.where(p_va - 0.5 > t, CALL, np.where(p_va - 0.5 < -t, PUT, NONE))
        trade = act != NONE
        k = int(trade.sum())
        if k < min_trades or k / n < min_cov:
            continue
        acc = float((act[trade] == np.where(yva[trade] == 1, CALL, PUT)).mean())
        if acc > best_info.get("val_acc", -1):
            best, best_info = t, {"threshold": t, "val_acc": acc, "val_trades": k, "val_coverage": k / n}
    return best, best_info


def cmd_stage3(args):
    from models import predict_p_call  # noqa: E402
    out = os.path.join(ROOT, "train", "out", args.segment, "trees")
    bpath = os.path.join(out, "bundle.joblib")
    if not os.path.exists(bpath):
        raise SystemExit("нет бандла: сначала stage2")
    bundle = joblib.load(bpath)
    pairs = bundle["pairs"]
    data = load_segment(args.segment, pairs)
    Xte, yte = data[2]
    p = predict_p_call(bundle["models"], Xte)
    thr = bundle["threshold"]
    pred = decide_arrays(p, thr, bundle["mode"])
    truth = np.where(yte == 1, CALL, PUT)
    m = summary(f"деревья ({bundle['mode']}) на ТЕСТЕ", pred, truth, CONFIG["payout"])
    print(f"тест (один раз): сегмент {args.segment}, примеров {len(yte)}")
    print("  " + fmt(m))
    # базовые линии на тесте, только для сравнения в отчёте
    Xtr, ytr = data[0]
    bl = baselines(Xtr, ytr, Xte, truth)
    bl_metrics = [summary(n, pr, truth, CONFIG["payout"]) for n, pr in bl.items()]
    for bm in bl_metrics:
        print("  " + fmt(bm))
    best_bl = max((b["acc"] for b in bl_metrics if b["n_trade"] > 0), default=0.0)
    ok, reasons = gate(m, best_bl, CONFIG["segments"][args.segment]["min_test_trades"])
    log = os.path.join(out, "test_log.jsonl")
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "passed": ok,
                            "mode": bundle["mode"], "n_trade": m["n_trade"], "acc": m["acc"],
                            "ci_low": m["ci95"][0], "bundle_sha256": sha256(bpath)}, ensure_ascii=False) + "\n")
    with open(log, encoding="utf-8") as f:
        uses = sum(1 for _ in f)
    print(f"\nкритерий допуска: {'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'}")
    for r in reasons:
        print("  - " + r)
    print(f"тест использован раз: {uses}")
    if args.approve:
        if not ok:
            raise SystemExit("одобрение отклонено: критерий не пройден")
        with open(os.path.join(out, "APPROVED.json"), "w", encoding="utf-8") as f:
            json.dump({"segment": args.segment, "bundle_sha256": sha256(bpath), "threshold": thr,
                       "mode": bundle["mode"], "test_acc": m["acc"], "test_ci_low": m["ci95"][0],
                       "test_trades": m["n_trade"], "approved_at": datetime.now(timezone.utc).isoformat()},
                      f, ensure_ascii=False, indent=2)
        print("сегмент одобрен: APPROVED.json записан")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--csv", required=True)
    p.add_argument("--pair", required=True); p.add_argument("--segment", default="otc")
    p = sub.add_parser("stage1"); p.add_argument("--pair", required=True); p.add_argument("--segment", default="otc")
    p = sub.add_parser("stage2"); p.add_argument("--segment", required=True, choices=["currency", "otc"])
    p.add_argument("--mode", default="always", choices=["always", "abstain"])
    p.add_argument("--models", default="rf"); p.add_argument("--pairs", help="через запятую; по умолчанию все подготовленные")
    p.add_argument("--rf-trees", type=int, default=100); p.add_argument("--max-train-rows", type=int, default=0)
    p.add_argument("--min-val-trades", type=int, default=300); p.add_argument("--min-coverage", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("stage3"); p.add_argument("--segment", required=True, choices=["currency", "otc"])
    p.add_argument("--approve", action="store_true")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "stage1": cmd_stage1, "stage2": cmd_stage2, "stage3": cmd_stage3}[args.cmd](args)


if __name__ == "__main__":
    main()
