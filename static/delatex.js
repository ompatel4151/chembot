// Convert stray LaTeX (some models write equations this way) into plain, readable text.
// Shared by the student chat and the batch questions page.
const SUPERSCRIPT = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻', '−': '⁻', '+': '⁺' };
const sup = t => [...t].map(c => SUPERSCRIPT[c] || c).join('');

function deLatex(s) {
  return (s || '')
    .replace(/\\[\[\]]/g, '')                                      // \[ \] display-math markers
    .replace(/\\?text\s*\{([^}]*)\}/g, '$1')
    .replace(/\\?mathrm\s*\{([^}]*)\}/g, '$1')
    .replace(/\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '($1)/($2)')
    .replace(/\\sqrt\s*\{([^{}]*)\}/g, '√($1)')
    .replace(/\\boxed\s*\{([^{}]*)\}/g, '$1')
    .replace(/\\(?:longrightarrow|rightarrow|to)\b/g, ' → ')
    .replace(/\\rightleftharpoons\b/g, ' ⇌ ')
    .replace(/\\leftarrow\b/g, ' ← ')
    .replace(/\\Delta\b/g, 'Δ').replace(/\\times\b/g, '×').replace(/\\cdot\b/g, '·')
    .replace(/\\pm\b/g, '±').replace(/\\leq\b/g, '≤').replace(/\\geq\b/g, '≥')
    .replace(/\\approx\b/g, '≈').replace(/\\neq\b/g, '≠')
    .replace(/\^\{([-−+]?\d+)\}/g, (m, t) => sup(t)).replace(/\^(\d+)/g, (m, t) => sup(t))
    .replace(/\^\{([^}]*)\}/g, '$1').replace(/_\{([^}]*)\}/g, '$1')
    .replace(/_(\d)/g, '$1')
    .replace(/\\(?:left|right|big|Big|!|\(|\)|\[|\])/g, '')
    .replace(/\\[ ,;:]/g, ' ')                                     // LaTeX spaces
    .replace(/\\[a-zA-Z]+/g, '')   // drop any remaining \command
    .replace(/[$]/g, '');          // strip dollar-sign math delimiters
}

if (typeof module !== 'undefined') module.exports = deLatex;   // lets test_delatex.js load it
