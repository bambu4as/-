/**
 * ml_indicators.js — основной набор индикаторов для модели (ind_main8).
 *
 * Формулы совпадают с analyzer.js (calculateIndicators), но значения НЕ округляются.
 * Считаются по последним IND_WINDOW закрытым барам. Python-двойник: ml/indicators.py.
 * Совпадение проверяется тестом по фикстуре tests/fixtures/ind_parity.json.
 */
export const IND_WINDOW = 300;
export const IND_FEATURES = ['rsi7', 'rsi14', 'stoch14', 'macd_hist', 'atr_pct14', 'bb_pos20', 'bb_z20', 'last_move_atr'];
export const IND_SET_NAME = 'ind_main8';

function rsiLast(c, period) {
    const n = c.length;
    if (n < period + 1) return 50;
    let gains = 0, losses = 0;
    for (let i = 1; i <= period; i++) {
        const d = c[i] - c[i - 1];
        if (d >= 0) gains += d; else losses -= d;
    }
    let ag = gains / period, al = losses / period;
    let value = 50;
    const val = () => (ag === 0 && al === 0) ? 50 : (al === 0) ? 100 : (ag === 0) ? 0 : 100 - 100 / (1 + ag / al);
    value = val();
    for (let i = period + 1; i < n; i++) {
        const d = c[i] - c[i - 1];
        const gain = d >= 0 ? d : 0;
        const loss = d < 0 ? -d : 0;
        ag = (ag * (period - 1) + gain) / period;
        al = (al * (period - 1) + loss) / period;
        value = val();
    }
    return value;
}

function macdHistLast(c) {
    const n = c.length;
    if (n < 35) return null;
    const k12 = 2 / 13, k26 = 2 / 27, kSig = 2 / 10;
    let emaF = c.slice(0, 12).reduce((a, b) => a + b, 0) / 12;
    let emaS = c.slice(0, 26).reduce((a, b) => a + b, 0) / 26;
    const macd = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
        if (i >= 12) emaF = c[i] * k12 + emaF * (1 - k12);
        if (i >= 26) {
            emaS = c[i] * k26 + emaS * (1 - k26);
            macd[i] = emaF - emaS;
        }
    }
    let sig = 0;
    for (let i = 26; i < 35; i++) sig += macd[i];
    sig /= 9;
    let hist = null;
    for (let i = 35; i < n; i++) {
        sig = macd[i] * kSig + sig * (1 - kSig);
        hist = macd[i] - sig;
    }
    return hist;
}

function atrLast(h, l, c) {
    const n = c.length;
    const tr = new Array(n).fill(0);
    for (let i = 1; i < n; i++) tr[i] = Math.max(h[i] - l[i], Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1]));
    const p = Math.min(14, n - 1);
    if (p <= 0) return 0;
    let atr = 0;
    for (let i = 1; i <= p; i++) atr += tr[i];
    atr /= p;
    for (let i = p + 1; i < n; i++) atr = (atr * (p - 1) + tr[i]) / p;
    return atr;
}

/** Основные индикаторы по последним IND_WINDOW барам. null, если данных меньше 35 баров. */
export function mainIndicators(closedBars) {
    const bars = closedBars.slice(-IND_WINDOW);
    const n = bars.length;
    if (n < 35) return null;
    const c = bars.map(b => Number(b.close));
    const h = bars.map(b => Number(b.high));
    const l = bars.map(b => Number(b.low));
    const last = c[n - 1];

    const rsi7 = rsiLast(c, 7);
    const rsi14 = rsiLast(c, 14);

    const hi14 = Math.max(...h.slice(-14));
    const lo14 = Math.min(...l.slice(-14));
    const stoch14 = hi14 === lo14 ? 50 : ((last - lo14) / (hi14 - lo14)) * 100;

    const macdHist = macdHistLast(c);

    const atr = atrLast(h, l, c);
    const atrPct14 = (atr / last) * 100;
    const lastMoveAtr = atr > 0 ? (c[n - 1] - c[n - 2]) / atr : 0;

    const bb = c.slice(-Math.min(20, n));
    const mean = bb.reduce((a, b) => a + b, 0) / bb.length;
    const sd = bb.length > 1 ? Math.sqrt(bb.reduce((s, v) => s + (v - mean) ** 2, 0) / bb.length) : 0;
    const upper = mean + 2 * sd, lower = mean - 2 * sd;
    const bbPos = upper === lower ? 0.5 : (last - lower) / (upper - lower);
    const bbZ = sd > 0 ? (last - mean) / sd : 0;

    return {
        rsi7, rsi14, stoch14,
        macd_hist: macdHist,
        atr_pct14: atrPct14,
        bb_pos20: bbPos,
        bb_z20: bbZ,
        last_move_atr: lastMoveAtr,
    };
}
