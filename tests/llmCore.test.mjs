import { test, before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { decideCore, buildRequest, validateDecision, coreMode } from '../llmCore.js';

// Синтетические свечи с ростом
const bars = Array.from({ length: 60 }, (_, i) => ({
    time: 1700000000 + i * 60,
    open: 1 + i * 0.0001,
    high: 1 + i * 0.0001 + 0.00005,
    low: 1 + i * 0.0001 - 0.00005,
    close: 1 + i * 0.0001 + (i % 2 ? 0.00002 : -0.00002)
}));

let server;
let port;
let nextReply = null;     // объект | 'timeout' | 'http500'

before(async () => {
    server = http.createServer((req, res) => {
        let body = '';
        req.on('data', c => (body += c));
        req.on('end', () => {
            if (nextReply === 'http500') { res.writeHead(500); return res.end('boom'); }
            if (nextReply === 'http409') { res.writeHead(409); return res.end('mismatch'); }
            if (nextReply === 'timeout') { return; }
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(nextReply));
        });
    });
    await new Promise(r => server.listen(0, '127.0.0.1', r));
    port = server.address().port;
});

after(() => server.close());

beforeEach(() => {
    process.env.LLM_URL = `http://127.0.0.1:${port}/decide`;
    process.env.LLM_TIMEOUT_MS = '300';
    delete process.env.LLM_FALLBACK;
    nextReply = { ok: true, action: 'CALL', confidence: 0.6, reason: 'test' };
});

test('coreMode всегда llm', () => {
    assert.equal(coreMode(), 'llm');
});

test('buildRequest: доходности в б.п., ограничение длины', () => {
    const req = buildRequest(bars, { pair: 'EUR/USD OTC', tf: 1 });
    assert.equal(req.returns_bp.length, 20);
    assert.ok(req.returns_bp.every(Number.isInteger));
});

test('validateDecision: отбрасывает неверные ответы', () => {
    assert.equal(validateDecision({ ok: true, action: 'BUY' }).ok, false);
    assert.equal(validateDecision({ ok: false, error: 'no_json' }).ok, false);
    assert.equal(validateDecision(null).ok, false);
    const v = validateDecision({ ok: true, action: 'put', confidence: 9, reason: 'x' });
    assert.equal(v.action, 'PUT');
    assert.equal(v.confidence, 1);
});

test('decideCore: берёт решение модели', async () => {
    nextReply = { ok: true, action: 'PUT', confidence: 0.42, reason: 'overbought' };
    const d = await decideCore(bars, { pair: 'X', tf: 1 });
    assert.equal(d.source, 'llm');
    assert.equal(d.action, 'PUT');
    assert.equal(d.confidence, 0.42);
    assert.equal(d.reason, 'overbought');
});

test('decideCore: NONE от модели уважается (сделки нет)', async () => {
    nextReply = { ok: true, action: 'NONE', confidence: 0.1, reason: 'mixed' };
    const d = await decideCore(bars, { pair: 'X', tf: 1 });
    assert.equal(d.action, 'NONE');
    assert.equal(d.source, 'llm');
});

test('decideCore: HTTP 500 -> NONE, сделки нет', async () => {
    nextReply = 'http500';
    const d = await decideCore(bars, { pair: 'X', tf: 1 });
    assert.equal(d.action, 'NONE');
    assert.equal(d.source, 'fallback-none');
    assert.match(d.llmError, /HTTP 500/);
});

test('decideCore: таймаут считается сбоем, а не зависанием', async () => {
    nextReply = 'timeout';
    const t0 = Date.now();
    const d = await decideCore(bars, { pair: 'X', tf: 1 });
    assert.ok(Date.now() - t0 < 2000);
    assert.equal(d.action, 'NONE');
    assert.equal(d.llmError, 'timeout');
});

test('decideCore: сервис недоступен -> NONE, а не исключение', async () => {
    process.env.LLM_URL = 'http://127.0.0.1:1/decide';
    const d = await decideCore(bars, { pair: 'X', tf: 1 });
    assert.equal(d.action, 'NONE');
    assert.match(d.llmError, /unreachable/);
});

test('decideCore: мусор в ответе -> NONE', async () => {
    nextReply = { ok: true, action: 'HOLD' };
    const d = await decideCore(bars, { pair: 'X', tf: 1 });
    assert.equal(d.action, 'NONE');
    assert.equal(d.llmError, 'bad_action');
});

import { segmentOf as segOf, segmentConfig, CONFIG } from '../config/index.js';
import { llmUrlFor } from '../llmCore.js';

test('segmentOf: сегмент только по имени, без зависимости от дня недели', () => {
    assert.equal(segOf('EUR/USD'), 'currency');
    assert.equal(segOf('EUR/USD OTC'), 'otc');
    assert.equal(segOf('usd/jpy otc'), 'otc');
    assert.equal(segOf(''), 'currency');
});

test('segmentConfig: все сегменты из конфига валидны, неизвестный отвергается', () => {
    for (const s of Object.keys(CONFIG.segments)) assert.ok(segmentConfig(s));
    assert.throws(() => segmentConfig('forex'));
});

test('llmUrlFor: сегментный адрес, затем общий LLM_URL', () => {
    process.env.LLM_URL_OTC = 'http://127.0.0.1:9001/decide';
    delete process.env.LLM_URL_CURRENCY;
    process.env.LLM_URL = 'http://127.0.0.1:9000/decide';
    assert.equal(llmUrlFor('otc'), 'http://127.0.0.1:9001/decide');
    assert.equal(llmUrlFor('currency'), 'http://127.0.0.1:9000/decide');
    delete process.env.LLM_URL_OTC;
    process.env.LLM_URL = `http://127.0.0.1:${port}/decide`;
});

test('buildRequest передаёт сегмент', () => {
    assert.equal(buildRequest(bars, { pair: 'EUR/USD OTC', tf: 1 }).segment, 'otc');
    assert.equal(buildRequest(bars, { pair: 'EUR/USD', tf: 1 }).segment, 'currency');
});

test('decideCore: решение несёт сегмент, ответ модели принимается', async () => {
    nextReply = { ok: true, action: 'CALL', confidence: 0.7, reason: 'x' };
    const d = await decideCore(bars, { pair: 'EUR/USD OTC', tf: 1 });
    assert.equal(d.segment, 'otc');
    assert.equal(d.source, 'llm');
});

test('decideCore: отказ сервиса на чужой сегмент (409) -> NONE', async () => {
    nextReply = 'http409';
    const d = await decideCore(bars, { pair: 'EUR/USD', tf: 1 });
    assert.equal(d.action, 'NONE');
    assert.match(d.llmError, /HTTP 409/);
});

import { tfLabel, tfText, isSupportedTf, TIMEFRAMES_SEC } from '../timeframes.js';

test('таймфреймы: поддерживаемые значения и метки', () => {
    assert.deepEqual(TIMEFRAMES_SEC, [5, 15, 30, 60, 180, 300]);
    assert.equal(isSupportedTf(5), true);
    assert.equal(isSupportedTf(1), false);        // старые минуты больше не принимаются
    assert.equal(isSupportedTf(120), false);
    assert.equal(tfLabel(5), '5s');
    assert.equal(tfLabel(60), '1m');
    assert.equal(tfLabel(300), '5m');
    assert.equal(tfText(15), '15 с');
    assert.equal(tfText(180), '3 хв');
});

test('llmUrlFor: адрес по периоду, затем по сегменту, затем общий', () => {
    delete process.env.LLM_URL_OTC; delete process.env.LLM_URL_OTC_5;
    process.env.LLM_URL = 'http://common/decide';
    process.env.LLM_URL_OTC = 'http://otc-1m/decide';
    process.env.LLM_URL_OTC_5 = 'http://otc-5s/decide';
    assert.equal(llmUrlFor('otc', 5), 'http://otc-5s/decide');
    assert.equal(llmUrlFor('otc', 60), 'http://otc-1m/decide');
    assert.equal(llmUrlFor('otc', 15), 'http://otc-1m/decide');   // нет 15 с — берём сегментный
    delete process.env.LLM_URL_OTC; delete process.env.LLM_URL_OTC_5; delete process.env.LLM_URL;
    process.env.LLM_URL = `http://127.0.0.1:${port}/decide`;
});

import { mainIndicators as _mi } from '../ml_indicators.js';

test('набор returns20: запрос без индикаторов', () => {
    delete process.env.LLM_FEATURE_SET;
    const req = buildRequest(bars, { pair: 'EUR/USD OTC', tf: 60 });
    assert.equal(req.feature_set, 'returns20');
    assert.equal('indicators' in req, false);
});

test('набор returns20+ind_main8: ровно 8 индикаторов, без округления', () => {
    process.env.LLM_FEATURE_SET = 'returns20+ind_main8';
    try {
        const req = buildRequest(bars, { pair: 'EUR/USD OTC', tf: 60 });
        assert.equal(req.feature_set, 'returns20+ind_main8');
        assert.deepEqual(Object.keys(req.indicators).sort(),
            ['atr_pct14', 'bb_pos20', 'bb_z20', 'last_move_atr', 'macd_hist', 'rsi14', 'rsi7', 'stoch14']);
        const ref = _mi(bars);
        assert.equal(req.indicators.rsi14, ref.rsi14);      // без округления
    } finally {
        delete process.env.LLM_FEATURE_SET;
    }
});

test('неизвестный набор признаков: ошибка конфигурации', () => {
    process.env.LLM_FEATURE_SET = 'bogus';
    try {
        assert.throws(() => buildRequest(bars, { pair: 'X', tf: 60 }));
    } finally {
        delete process.env.LLM_FEATURE_SET;
    }
});
