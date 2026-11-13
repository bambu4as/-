import hashlib
import json
import os
import random
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from features import FEATURE_LEN, decide, features, returns_from_prompt  # noqa: E402
from common import make_prompt, returns_bp  # noqa: E402
from train_trees import pick_threshold  # noqa: E402
from tree_server import Service, make_server  # noqa: E402


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tiny_bundle(path, segment="otc", seed=0):
    from sklearn.ensemble import RandomForestClassifier
    rnd = np.random.RandomState(seed)
    X = rnd.normal(0, 10, size=(400, FEATURE_LEN))
    y = (X[:, -1] > 0).astype(int)
    m = RandomForestClassifier(n_estimators=20, min_samples_leaf=20, random_state=seed).fit(X, y)
    joblib.dump({"segment": segment, "models": {"rf": m}, "threshold": 0.02,
                 "feature_version": "returns20-v1", "model_names": ["rf"]}, path)


def approve(dir_, segment, bundle):
    with open(os.path.join(dir_, "APPROVED.json"), "w") as f:
        json.dump({"segment": segment, "bundle_sha256": sha(bundle)}, f)


class Features(unittest.TestCase):
    def test_prompt_parity_with_bp_returns(self):
        closes = [1.0 + 0.0003 * i * (-1) ** i for i in range(21)]
        p = make_prompt("EUR/USD OTC", 60, closes)
        self.assertEqual(returns_from_prompt(p), [float(x) for x in returns_bp(closes)])

    def test_incomplete_window_is_none(self):
        self.assertIsNone(returns_from_prompt("Recent 1-minute returns (bp), oldest to newest: none"))

    def test_features_validation(self):
        with self.assertRaises(ValueError):
            features([1, 2, 3])
        with self.assertRaises(ValueError):
            features(["a"] * FEATURE_LEN)
        self.assertEqual(len(features(list(range(FEATURE_LEN)))), FEATURE_LEN)


class Decision(unittest.TestCase):
    def test_mapping_and_abstain(self):
        self.assertEqual(decide(0.60, 0.05)[0], "CALL")
        self.assertEqual(decide(0.40, 0.05)[0], "PUT")
        self.assertEqual(decide(0.52, 0.05)[0], "NONE")     # в зоне воздержания
        self.assertEqual(decide(0.50, 0.0)[0], "NONE")               # с воздержанием ничья -> NONE
        self.assertEqual(decide(0.50, 0.0, tie="CALL")[0], "CALL")   # режим always: ничья -> CALL
        self.assertEqual(decide(0.49, 0.0, tie="CALL")[0], "PUT")    # порог 0: только CALL/PUT
        self.assertEqual(decide(0.5001, 0.0, tie="CALL")[0], "CALL")
        self.assertLessEqual(decide(0.99, 0.0)[1], 1.0)


class Importer(unittest.TestCase):
    def test_json_ms_and_dedupe(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from import_history import normalize
        rows = [{"time": 1_700_000_000_000, "open": 1, "high": 1, "low": 1, "close": 1},
                {"time": 1_700_000_060_000, "open": 1, "high": 2, "low": 1, "close": 2},
                {"time": 1_700_000_060_000, "open": 9, "high": 9, "low": 9, "close": 9},
                {"time": 1_700_000_120, "open": "x", "high": 1, "low": 1, "close": 1}]
        bars, skipped = normalize(rows)
        self.assertEqual(sorted(bars), [1_700_000_000, 1_700_000_060])
        self.assertEqual(bars[1_700_000_060][3], 9.0)        # дубликат: последний побеждает
        self.assertEqual(skipped, 1)


class Threshold(unittest.TestCase):
    def test_picks_threshold_on_planted_signal(self):
        rnd = random.Random(1)
        p, y = [], []
        for _ in range(2000):
            edge = rnd.uniform(-0.3, 0.3)
            p.append(0.5 + edge)
            y.append(1 if (edge > 0) == (rnd.random() < 0.8) else 0)   # правда в 80% случаев
        best = pick_threshold(np.array(p), np.array(y), min_trades=100, min_cov=0.1)
        self.assertIsNotNone(best)
        self.assertGreater(best["val_acc"], 0.7)

    def test_none_when_too_few_trades(self):
        p = np.array([0.5] * 50)
        y = np.array([1] * 50)
        self.assertIsNone(pick_threshold(p, y, min_trades=300, min_cov=0.2))


class ServiceApproval(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.bundle = os.path.join(self.dir, "bundle.joblib")
        tiny_bundle(self.bundle)

    def test_not_approved_refuses_to_trade(self):
        svc = Service("otc", bundle_path=self.bundle, approval_dir=self.dir)
        self.assertFalse(svc.approved)
        code, body = svc.decide({"segment": "otc", "returns_bp": [1] * FEATURE_LEN})
        self.assertEqual(code, 503)
        self.assertEqual(body["action"], "NONE")

    def test_approved_serves(self):
        approve(self.dir, "otc", self.bundle)
        svc = Service("otc", bundle_path=self.bundle, approval_dir=self.dir)
        self.assertTrue(svc.approved)
        code, body = svc.decide({"segment": "otc", "returns_bp": list(range(FEATURE_LEN))})
        self.assertEqual(code, 200)
        self.assertIn(body["action"], ("CALL", "PUT", "NONE"))

    def test_bundle_swap_invalidates_approval(self):
        approve(self.dir, "otc", self.bundle)
        tiny_bundle(self.bundle, seed=7)                 # подмена модели после одобрения
        svc = Service("otc", bundle_path=self.bundle, approval_dir=self.dir)
        self.assertFalse(svc.approved)
        self.assertEqual(svc.reason, "approval_mismatch")

    def test_wrong_segment_refused(self):
        approve(self.dir, "otc", self.bundle)
        svc = Service("otc", bundle_path=self.bundle, approval_dir=self.dir)
        code, body = svc.decide({"segment": "currency", "returns_bp": [1] * FEATURE_LEN})
        self.assertEqual((code, body["error"]), (409, "segment_mismatch"))

    def test_bad_input(self):
        approve(self.dir, "otc", self.bundle)
        svc = Service("otc", bundle_path=self.bundle, approval_dir=self.dir)
        code, _ = svc.decide({"segment": "otc", "returns_bp": [1, 2]})
        self.assertEqual(code, 400)


class HttpRoundTrip(unittest.TestCase):
    def test_decide_over_http(self):
        d = tempfile.mkdtemp()
        b = os.path.join(d, "bundle.joblib")
        tiny_bundle(b)
        approve(d, "otc", b)
        svc = Service("otc", bundle_path=b, approval_dir=d)
        srv = make_server(svc, port=0)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            port = srv.server_address[1]
            req = urllib.request.Request(f"http://127.0.0.1:{port}/decide",
                                         data=json.dumps({"segment": "otc", "returns_bp": [3] * FEATURE_LEN}).encode(),
                                         method="POST")
            with urllib.request.urlopen(req) as r:
                body = json.loads(r.read())
            self.assertTrue(body["ok"])
            self.assertIn(body["action"], ("CALL", "PUT", "NONE"))
            req2 = urllib.request.Request(f"http://127.0.0.1:{port}/decide",
                                          data=json.dumps({"segment": "currency", "returns_bp": [3] * FEATURE_LEN}).encode(),
                                          method="POST")
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req2)
            self.assertEqual(cm.exception.code, 409)
        finally:
            srv.shutdown()
            srv.server_close()



class NumericVsJsonParity(unittest.TestCase):
    """Числовой путь (prepare.py) и JSON-путь (build_dataset.py) дают одинаковые примеры."""

    def test_same_times_labels_features(self):
        import csv
        import tempfile
        import build_dataset as bd
        import prepare as pp
        rnd = random.Random(3)
        n, t0, c = 1500, 1_700_000_000, 1.0
        d = tempfile.mkdtemp()
        path = os.path.join(d, "hist_TST_USD_OTC_1m.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "open", "high", "low", "close"])
            for i in range(n):
                o = c
                c = round(c * (1 + rnd.gauss(0, 2e-4)), 6)
                w.writerow([t0 + i * 60, o, max(o, c), min(o, c), c])
        times, closes = pp.load_closes(path)
        X, y, t, s, _ = pp.build_pair(times, closes)
        ts, cs = bd.load_bars(path)
        splits, _ = bd.build(ts, cs, 60, "TST/USD OTC", 60)
        rows = sorted((r for k in ("train", "val", "test") for r in splits[k]), key=lambda r: r["t"])
        self.assertEqual(len(rows), len(y))
        o = np.argsort(t)
        self.assertTrue(np.array_equal(t[o], np.array([r["t"] for r in rows])))
        self.assertTrue(np.array_equal(y[o], np.array([1 if r["action"] == "CALL" else 0 for r in rows])))
        jx = np.array([returns_from_prompt(r["prompt"]) for r in rows], dtype=np.float32)
        self.assertTrue(np.allclose(X[o], jx))


class ExtraFeaturesNoLookahead(unittest.TestCase):
    """Расширенные признаки на баре i не зависят от баров после i."""

    def test_no_future_leak(self):
        import prepare as pp
        rnd = random.Random(9)
        n, t0 = 400, 1_700_000_000
        t = np.array([t0 + k * 60 for k in range(n)], dtype=np.int64)
        c = np.cumprod(1 + np.array([rnd.gauss(0, 2e-4) for _ in range(n)])) + 1.0
        o = c * (1 + 1e-5); h = c * (1 + 2e-4); l = c * (1 - 2e-4)
        i = 200
        base = pp.extra_features(t, o, h, l, c, i)
        c2, h2, l2, o2 = c.copy(), h.copy(), l.copy(), o.copy()
        c2[i + 1:] *= 1.05; h2[i + 1:] *= 1.05; l2[i + 1:] *= 0.95; o2[i + 1:] *= 1.02
        after = pp.extra_features(t, o2, h2, l2, c2, i)
        self.assertTrue(np.allclose(np.concatenate([np.ravel(v) for v in base]),
                                    np.concatenate([np.ravel(v) for v in after])))

    def test_build_pair_ext_keeps_returns(self):
        import csv
        import tempfile
        import prepare as pp
        rnd = random.Random(4)
        n, t0, c = 1200, 1_700_000_000, 1.0
        d = tempfile.mkdtemp()
        path = os.path.join(d, "hist_TST_USD_OTC_1m.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "open", "high", "low", "close"])
            for k in range(n):
                o_ = c
                c = round(c * (1 + rnd.gauss(0, 2e-4)), 6)
                w.writerow([t0 + k * 60, o_, max(o_, c) * 1.0001, min(o_, c) * 0.9999, c])
        t, o_, h_, l_, c_ = pp.load_ohlc(path)
        X1, y1, t1, s1, _ = pp.build_pair(t, c_)
        X2, E2, y2, t2, s2 = pp.build_pair_ext(t, o_, h_, l_, c_)
        self.assertTrue(np.array_equal(X1, X2))
        self.assertTrue(np.array_equal(y1, y2))
        self.assertEqual(E2.shape, (len(y2), 25))
        self.assertTrue(np.all(E2[:, 0:20] >= 0))            # диапазон не отрицательный


class IndicatorParity(unittest.TestCase):
    """Python-индикаторы (полная история) совпадают с Node (окно 300 баров) по фикстуре."""

    def test_matches_js_fixture(self):
        import json as _json
        from indicators import IND_FEATURES, main_indicators
        fx = _json.load(open(os.path.join(ROOT, "tests", "fixtures", "ind_parity.json"), encoding="utf-8"))
        bars = fx["bars"]
        h = np.array([b["high"] for b in bars]); l = np.array([b["low"] for b in bars])
        c = np.array([b["close"] for b in bars])
        full = main_indicators(h, l, c)
        worst = 0.0
        for e in fx["expected"]:
            for k in IND_FEATURES:
                a, b = float(full[k][e["pos"]]), float(e["values"][k])
                worst = max(worst, abs(a - b) / max(1.0, abs(b)))
        self.assertLess(worst, 1e-6, f"худшее отклонение {worst}")


class FeatureSets(unittest.TestCase):
    def ind(self, v=1.0):
        from features import IND_FEATURES
        return {k: v for k in IND_FEATURES}

    def test_returns_only_ignores_indicators(self):
        from features import features
        self.assertEqual(len(features([1] * FEATURE_LEN, None, "returns20")), FEATURE_LEN)

    def test_ind_set_needs_all_eight(self):
        from features import features
        vec = features([1] * FEATURE_LEN, self.ind(2.0), "returns20+ind_main8")
        self.assertEqual(len(vec), FEATURE_LEN + 8)
        with self.assertRaises(ValueError):
            features([1] * FEATURE_LEN, None, "returns20+ind_main8")
        bad = self.ind(1.0); bad.pop("rsi14")
        with self.assertRaises(ValueError):
            features([1] * FEATURE_LEN, bad, "returns20+ind_main8")
        with self.assertRaises(ValueError):
            features([1] * FEATURE_LEN, self.ind(float("nan")), "returns20+ind_main8")

    def test_service_refuses_other_feature_set(self):
        d = tempfile.mkdtemp()
        b = os.path.join(d, "bundle.joblib")
        tiny_bundle(b)
        approve(d, "otc", b)
        svc = Service("otc", bundle_path=b, approval_dir=d, feature_set="returns20")
        code, body = svc.decide({"segment": "otc", "feature_set": "returns20+ind_main8",
                                 "returns_bp": [1] * FEATURE_LEN, "indicators": self.ind()})
        self.assertEqual((code, body["error"]), (409, "feature_set_mismatch"))

    def test_bundle_feature_set_must_match_service(self):
        d = tempfile.mkdtemp()
        b = os.path.join(d, "bundle.joblib")
        tiny_bundle(b)                                  # набор по умолчанию: returns20
        approve(d, "otc", b)
        svc = Service("otc", bundle_path=b, approval_dir=d, feature_set="returns20+ind_main8")
        code, body = svc.decide({"segment": "otc", "feature_set": "returns20+ind_main8",
                                 "returns_bp": [1] * FEATURE_LEN, "indicators": self.ind()})
        self.assertEqual((code, body["error"]), (503, "bundle_feature_set_mismatch"))


if __name__ == "__main__":
    unittest.main()
