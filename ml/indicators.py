"""
indicators.py — основной набор индикаторов ind_main8 (двойник ml_indicators.js).

Формулы те же, что в JavaScript. Здесь значения считаются по ВСЕЙ истории одним проходом (для обучения
и быстро), а в работе — по последним 300 барам. Расхождение из-за начальной точки EMA/Wilder
затухает ниже 1e-6 за 300 баров; это проверяет тест по фикстуре tests/fixtures/ind_parity.json.
"""
import numpy as np

IND_SET_NAME = "ind_main8"
IND_FEATURES = ["rsi7", "rsi14", "stoch14", "macd_hist", "atr_pct14", "bb_pos20", "bb_z20", "last_move_atr"]


def _rsi_series(c, p):
    n = len(c)
    out = np.full(n, np.nan)
    if n < p + 1:
        return out
    d = np.diff(c[:p + 1])
    ag = np.sum(np.where(d >= 0, d, 0.0)) / p
    al = np.sum(np.where(d < 0, -d, 0.0)) / p

    def val(ag, al):
        if ag == 0 and al == 0:
            return 50.0
        if al == 0:
            return 100.0
        if ag == 0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    out[p] = val(ag, al)
    for i in range(p + 1, n):
        dd = c[i] - c[i - 1]
        g = dd if dd >= 0 else 0.0
        l_ = -dd if dd < 0 else 0.0
        ag = (ag * (p - 1) + g) / p
        al = (al * (p - 1) + l_) / p
        out[i] = val(ag, al)
    return out


def _macd_hist_series(c):
    n = len(c)
    out = np.full(n, np.nan)
    if n < 35:
        return out
    k12, k26, kSig = 2 / 13, 2 / 27, 2 / 10
    emaF = c[:12].mean()
    emaS = c[:26].mean()
    macd = np.full(n, np.nan)
    for i in range(n):
        if i >= 12:
            emaF = c[i] * k12 + emaF * (1 - k12)
        if i >= 26:
            emaS = c[i] * k26 + emaS * (1 - k26)
            macd[i] = emaF - emaS
    sig = np.sum(macd[26:35]) / 9
    for i in range(35, n):
        sig = macd[i] * kSig + sig * (1 - kSig)
        out[i] = macd[i] - sig
    return out


def _atr_pct_series(h, l, c, p=14):
    n = len(c)
    out = np.full(n, np.nan)
    if n < p + 1:
        return out
    tr = np.zeros(n)
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = tr[1:p + 1].sum() / p
    out[p] = atr / c[p] * 100
    for i in range(p + 1, n):
        atr = (atr * (p - 1) + tr[i]) / p
        out[i] = atr / c[i] * 100
    return out


def _atr_value_series(h, l, c, p=14):
    """ATR в абсолютных единицах (нужен для last_move_atr)."""
    n = len(c)
    out = np.full(n, np.nan)
    if n < p + 1:
        return out
    tr = np.zeros(n)
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = tr[1:p + 1].sum() / p
    out[p] = atr
    for i in range(p + 1, n):
        atr = (atr * (p - 1) + tr[i]) / p
        out[i] = atr
    return out


def _sliding(a, w):
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(a, w)


def main_indicators(h, l, c):
    """Все 8 признаков на каждом баре (nan, пока данных не хватает). Массивы той же длины, что c."""
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    res = {k: np.full(n, np.nan) for k in IND_FEATURES}
    res["rsi7"] = _rsi_series(c, 7)
    res["rsi14"] = _rsi_series(c, 14)

    hi = np.full(n, np.nan); lo = np.full(n, np.nan)
    if n >= 14:
        hi[13:] = _sliding(h, 14).max(axis=1)
        lo[13:] = _sliding(l, 14).min(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stoch = np.where(hi == lo, 50.0, (c - lo) / (hi - lo) * 100.0)
    res["stoch14"] = stoch

    res["macd_hist"] = _macd_hist_series(c)
    res["atr_pct14"] = _atr_pct_series(h, l, c)

    mean = np.full(n, np.nan); sd = np.full(n, np.nan)
    if n >= 20:
        w = _sliding(c, 20)
        mean[19:] = w.mean(axis=1)
        sd[19:] = w.std(axis=1)                     # population sd, как в analyzer.js
    upper = mean + 2 * sd; lower = mean - 2 * sd
    with np.errstate(invalid="ignore", divide="ignore"):
        res["bb_pos20"] = np.where(upper == lower, 0.5, (c - lower) / (upper - lower))
        res["bb_z20"] = np.where(sd > 0, (c - mean) / sd, 0.0)

    atr_abs = _atr_value_series(h, l, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.zeros(n); move[1:] = c[1:] - c[:-1]
        res["last_move_atr"] = np.where(atr_abs > 0, move / atr_abs, 0.0)
    return res
