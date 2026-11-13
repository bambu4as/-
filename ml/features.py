"""
features.py — признаки и решение для деревьев. Версия признаков зафиксирована в бандле модели.

Признаки v1: 20 доходностей в б.п. (то же представление, что у LLM и в buildRequest() в llmCore.js).
Решение: P(CALL) — среднее по моделям; воздержание, если |P - 0.5| < порога.
"""
import re

import numpy as np

FEATURE_LEN = 20
IND_FEATURES = ["rsi7", "rsi14", "stoch14", "macd_hist", "atr_pct14", "bb_pos20", "bb_z20", "last_move_atr"]
# Наборы признаков. Имя набора сохраняется в бандле модели и проверяется сервисом на каждом запросе.
FEATURE_SETS = {
    "returns20": FEATURE_LEN,
    "returns20+ind_main8": FEATURE_LEN + len(IND_FEATURES),
}
DEFAULT_FEATURE_SET = "returns20"
FEATURE_VERSION = "returns20-v1"
_RE = re.compile(r"oldest to newest: ([^\n]*)")


def returns_from_prompt(prompt: str):
    """Доходности из промпта датасета. None, если окно неполное. Совпадает с train/evaluate.py:features_of."""
    m = _RE.search(prompt)
    if not m or m.group(1).strip() == "none":
        return None
    vals = [float(x) for x in m.group(1).split(",")]
    return vals if len(vals) == FEATURE_LEN else None


def features(returns_bp, indicators=None, feature_set=DEFAULT_FEATURE_SET):
    """
    Вектор признаков из запроса сервиса. Бросает ValueError при неверных данных.
    returns20: только доходности. returns20+ind_main8: доходности и 8 индикаторов (в порядке IND_FEATURES).
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"неизвестный набор признаков: {feature_set}")
    if not isinstance(returns_bp, list) or len(returns_bp) != FEATURE_LEN:
        raise ValueError(f"нужно {FEATURE_LEN} доходностей")
    try:
        vec = [float(x) for x in returns_bp]
    except (TypeError, ValueError):
        raise ValueError("доходности должны быть числами")
    if feature_set == "returns20+ind_main8":
        if not isinstance(indicators, dict):
            raise ValueError("нужны индикаторы ind_main8")
        for k in IND_FEATURES:
            v = indicators.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(float(v)):
                raise ValueError(f"индикатор {k} отсутствует или не число")
            vec.append(float(v))
    return vec


def decide(p_call: float, threshold: float, tie: str = "NONE"):
    """
    Направление по вероятности CALL.
    |P - 0.5| < threshold -> воздержание (NONE). confidence = |P - 0.5| * 2, в [0, 1].
    Ровно P = 0.5 (ничья голосов): ответ tie ("NONE" по умолчанию, "CALL" в режиме always).
    """
    edge = p_call - 0.5
    confidence = min(1.0, abs(edge) * 2)
    if edge == 0:
        return tie, confidence
    if abs(edge) < threshold:
        return "NONE", confidence
    return ("CALL" if edge > 0 else "PUT"), confidence
