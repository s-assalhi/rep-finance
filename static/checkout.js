/* YGZ SHOP checkout — laadt op elke pagina, opent iDEAL-checkout via /checkout */
(function () {
  if (window.__ygzCheckout) return;
  window.__ygzCheckout = true;

  var css = document.createElement('style');
  css.textContent = [
    '#ygz-pay-btn{position:fixed;right:18px;bottom:18px;z-index:99998;display:flex;align-items:center;gap:8px;',
    'padding:13px 22px;border:none;border-radius:999px;background:linear-gradient(135deg,#d63384,#ff6a3d);',
    'color:#fff;font-weight:700;font-size:15px;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.35);font-family:inherit}',
    '#ygz-pay-btn:hover{filter:brightness(1.08)}',
    '#ygz-pay-ov{position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,.65);display:none;',
    'align-items:center;justify-content:center;padding:20px;font-family:inherit}',
    '#ygz-pay-ov.open{display:flex}',
    '#ygz-pay-card{background:#fff;border-radius:16px;max-width:380px;width:100%;padding:24px;',
    'font-family:-apple-system,"Segoe UI",Roboto,sans-serif;color:#111}',
    '#ygz-pay-card h3{margin:0 0 4px;font-size:18px}',
    '#ygz-pay-card .sub{margin:0 0 16px;color:#666;font-size:13px}',
    '#ygz-pay-card .row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}',
    '#ygz-pay-card .chip{flex:1;min-width:70px;padding:10px 0;text-align:center;border:1.5px solid #ddd;',
    'border-radius:10px;cursor:pointer;font-weight:700;font-size:14px;background:#fff}',
    '#ygz-pay-card .chip.sel{border-color:#d63384;background:#fdf2f7}',
    '#ygz-pay-card input{width:100%;box-sizing:border-box;padding:11px;border:1.5px solid #ddd;',
    'border-radius:10px;font-size:15px;margin-bottom:14px}',
    '#ygz-pay-card button.go{width:100%;padding:13px;border:none;border-radius:10px;',
    'background:linear-gradient(135deg,#d63384,#ff6a3d);color:#fff;font-weight:700;font-size:15px;cursor:pointer}',
    '#ygz-pay-card .note{margin:10px 0 0;font-size:11.5px;color:#888;line-height:1.5;text-align:center}',
    '#ygz-pay-card .x{float:right;background:none;border:none;font-size:20px;color:#999;cursor:pointer;line-height:1}'
  ].join('');
  document.head.appendChild(css);

  var btn = document.createElement('button');
  btn.id = 'ygz-pay-btn';
  btn.innerHTML = '💳 Betalen met iDEAL';
  btn.addEventListener('click', function () { ov.classList.add('open'); });
  document.body.appendChild(btn);

  var ov = document.createElement('div');
  ov.id = 'ygz-pay-ov';
  ov.innerHTML = [
    '<div id="ygz-pay-card">',
    '  <button class="x" aria-label="sluiten">×</button>',
    '  <h3>Betaling</h3>',
    '  <p class="sub">Betaal veilig via iDEAL. Vul het afgesproken bedrag in euro\'s.</p>',
    '  <div class="row">',
    '    <div class="chip" data-a="25">€25</div>',
    '    <div class="chip" data-a="50">€50</div>',
    '    <div class="chip" data-a="100">€100</div>',
    '    <div class="chip" data-a="200">€200</div>',
    '  </div>',
    '  <input id="ygz-amt" type="number" min="10" step="0.01" placeholder="Of eigen bedrag, bijv. 74.50">',
    '  <button class="go">Verder naar iDEAL →</button>',
    '  <p class="note">Je gaat naar een beveiligde betaalpagina. Na betaling wordt je order direct verwerkt.</p>',
    '</div>'
  ].join('');
  document.body.appendChild(ov);

  var sel = null;
  ov.addEventListener('click', function (e) {
    if (e.target === ov) { ov.classList.remove('open'); return; }
    var chip = e.target.closest ? e.target.closest('.chip') : null;
    if (chip) {
      sel = chip.getAttribute('data-a');
      ov.querySelectorAll('.chip').forEach(function (c) { c.classList.remove('sel'); });
      chip.classList.add('sel');
      ov.querySelector('#ygz-amt').value = '';
      return;
    }
    if (e.target.classList.contains('x')) { ov.classList.remove('open'); return; }
    if (e.target.classList.contains('go')) {
      var custom = parseFloat(ov.querySelector('#ygz-amt').value);
      var amt = (!isNaN(custom) && custom >= 10) ? custom : (sel ? parseFloat(sel) : null);
      if (!amt || amt < 10) { alert('Vul een bedrag in van minimaal €10.'); return; }
      window.location.href = '/checkout?amount=' + encodeURIComponent(amt.toFixed(2));
    }
  });
})();
