import csv
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))

import arrays  # noqa: E402
import build_dataset as bd  # noqa: E402
from features import returns_from_prompt  # noqa: E402


def write_csv(path, times, closes):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close"])
        for t, c in zip(times, closes):
            w.writerow([t, c, c, c, c])


class ParityWithPromptBuilder(unittest.TestCase):
    """Массивы должны давать тот же набор примеров, что train/build_dataset.py."""

    def test_same_examples_same_split_same_features(self):
        rnd = random.Random(3)
        n = 900
        times = [1_700_000_000 + i * 60 for i in range(n)]
        closes = [1.0]
        for _ in range(n - 1):
            closes.append(closes[-1] * (1 + rnd.gauss(0, 2e-4)))
        # разрыв во времени, чтобы проверить отбрасывание окон
        times = times[:400] + [t + 300 for t in times[400:]]
        closes_list = list(closes)

        a = arrays.build(np_array(times), np_array(closes_list), 60)
        splits, _ = bd.build(times, closes_list, 60, "T", 60)

        names = {0: "train", 1: "val", 2: "test"}
        for code, name in names.items():
            rows = splits[name]
            mask = a["split"] == code
            self.assertEqual(int(mask.sum()), len(rows), f"число примеров в {name}")
            self.assertEqual(a["t"][mask].tolist(), [r["t"] for r in rows], f"время в {name}")
            self.assertEqual(a["y"][mask].tolist(), [1 if r["action"] == "CALL" else 0 for r in rows], f"метка в {name}")
            for x, r in zip(a["X"][mask], rows):
                self.assertEqual(x.tolist(), returns_from_prompt(r["prompt"]), f"признаки в {name}")

    def test_js_round_half_up(self):
        self.assertEqual(arrays.js_round(np_array([2.5, -2.5, -0.4]))[:2].tolist(), [3.0, -2.0])


def np_array(v):
    import numpy as np
    return np.array(v, dtype=np.float64)


if __name__ == "__main__":
    unittest.main()
