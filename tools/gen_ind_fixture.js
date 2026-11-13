// Генерирует tests/fixtures/ind_parity.json: эталонные значения mainIndicators на детерминированном ряду.
// Запуск: node tools/gen_ind_fixture.js
import fs from 'fs';
import { mainIndicators, IND_WINDOW } from '../ml_indicators.js';

let seed = 12345;
const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const N = 600, T0 = 1700000000;
const bars = [];
let c = 1.1;
for (let i = 0; i < N; i++) {
    const o = c;
    c = c * (1 + (rnd() - 0.5) * 6e-4);
    const h = Math.max(o, c) * (1 + rnd() * 2e-4);
    const l = Math.min(o, c) * (1 - rnd() * 2e-4);
    bars.push({ time: T0 + i * 60, open: o, high: h, low: l, close: c });
}
const positions = [40, 320, 450, 599];
const expected = positions.map(p => ({ pos: p, values: mainIndicators(bars.slice(0, p + 1)) }));
fs.mkdirSync('tests/fixtures', { recursive: true });
fs.writeFileSync('tests/fixtures/ind_parity.json', JSON.stringify({ window: IND_WINDOW, bars, expected }, null, 1));
console.log('фикстура записана:', positions.join(', '));
