// node test_delatex.js — checks static/delatex.js on real ChemBot answers that contained LaTeX
const assert = require('assert');
const deLatex = require('./static/delatex.js');

// Q87
assert.strictEqual(
  deLatex('\\text{moles I}_2 = 0.4235\\ \\text{mol CuCl}_2 \\times \\frac{1\\ \\text{mol I}_2}{2\\ \\text{mol CuCl}_2}'),
  'moles I2 = 0.4235 mol CuCl2 × (1 mol I2)/(2 mol CuCl2)');
assert.strictEqual(deLatex('\\boxed{0.212\\ \\text{mol I}_2}'), '0.212 mol I2');
// Q57
assert.strictEqual(deLatex('M_{\\text{NH}_3}=14.01\\;\\text{g mol}^{-1} \\approx 17.03'), 'MNH3=14.01 g mol⁻¹ ≈ 17.03');
// Q117
assert.strictEqual(deLatex('\\(q_{\\text{system}} = m \\times c \\times \\Delta T\\)'), 'qsystem = m × c × Δ T');
// Q103
assert.strictEqual(deLatex('6.022 \\times 10^{23}'), '6.022 × 10²³');
// plain answers are left alone
const plain = '- A catalyst lowers the activation energy (Ea).\nSource: Catalyst Reading.pdf';
assert.strictEqual(deLatex(plain), plain);
assert.strictEqual(deLatex('1.04 x 10^14'), '1.04 x 10¹⁴');
console.log('ok');
