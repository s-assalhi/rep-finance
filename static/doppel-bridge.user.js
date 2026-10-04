// ==UserScript==
// @name         Rep Agent â€” doppel bridge
// @namespace    repagent
// @version      1.0.0
// @description  Voert zoek-/QC-opdrachten van de Rep Agent bot uit op doppel.fit (gebruikt jouw ingelogde tab).
// @match        https://doppel.fit/*
// @run-at       document-idle
// @connect      rep-finance-wa.onrender.com
// @connect      rep-finance.onrender.com
// @connect      localhost
// @grant        none
// ==/UserScript==

/*
 * Installatie (eenmalig):
 *   1. Installeer Tampermonkey in Chrome.
 *   2. Nieuw userscript -> plak dit bestand -> opslaan.
 *   3. Zet je backend + toegangscode in de console op doppel.fit:
 *        localStorage.setItem('repagent_key', 'RF4KD-9XQ2-PT7M')
 *        localStorage.setItem('repagent_back', 'https://rep-finance-wa.onrender.com')
 *   4. Laat een doppel.fit-tab open staan.
 * De bot kan dan zoekopdrachten uitvoeren: hij bestuurt deze tab kort
 * (zoeken -> top-items openen -> QC-foto's rapen) en stuurt het resultaat
 * naar de bot, die het naar je WhatsApp/Telegram stuurt.
 */
(function () {
  'use strict';

  const BACK = localStorage.getItem('repagent_back') || 'https://rep-finance-wa.onrender.com';
  const KEY = 'RF4KD-9XQ2-PT7M';
  const HDRS = { 'Content-Type': 'application/json', 'X-Access-Code': KEY };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function postResult(body) {
    for (let i = 0; i < 3; i++) {
      try {
        const r = await fetch(BACK + '/whatsapp/source/result', {
          method: 'POST', headers: HDRS, body: JSON.stringify(body),
        });
        if (r.ok) return true;
      } catch (e) { /* opnieuw */ }
      await sleep(2000);
    }
    return false;
  }

  async function probeerCropFoto(src) {
    // Watermerk (doppel.wm, linksboven/rechtsboven) wegwerken: bovenste 10% wegcroppen.
    return new Promise((res) => {
      const im = new Image();
      im.crossOrigin = 'anonymous';
      im.onload = () => {
        try {
          const c = document.createElement('canvas');
          c.width = im.naturalWidth;
          c.height = Math.round(im.naturalHeight * 0.9);
          const ctx = c.getContext('2d');
          ctx.drawImage(im, 0, Math.round(im.naturalHeight * 0.1),
            im.naturalWidth, im.naturalHeight * 0.9, 0, 0, c.width, c.height);
          res(c.toDataURL('image/jpeg', 0.82));
        } catch (e) { res(null); } // canvas tainted -> geen crop
      };
      im.onerror = () => res(null);
      im.src = src;
    });
  }

  function verkoperUitTekst(t) {
    const m = t.match(/([A-Za-z0-9][A-Za-z0-9 '&.-]{3,60}?)\s+\d\.\d\s*\(/);
    return m ? m[1].trim() : null;
  }

  async function scrapeItemPage() {
    await sleep(2500); // QC-plaatjes laten laden
    const h1 = document.querySelector('h1');
    const txt = (document.body.innerText || '').replace(/\s+/g, ' ');
    const titel = ((h1 && h1.textContent) || txt.slice(0, 120)).trim().slice(0, 120);
    const prijs = (txt.match(/â‚¬\s?[0-9]+(?:[.,][0-9]{1,2})?/) || [null])[0];
    const fotos = [...new Set(Array.from(document.querySelectorAll('img[src*="cdn.doppel.fit"]'))
      .map((i) => i.src))].slice(0, 12);
    const item = {
      titel, prijs, url: location.href,
      verkoper: verkoperUitTekst(txt),
      fotos,
      foto_b64: null,
    };
    if (fotos.length) item.foto_b64 = await probeerCropFoto(fotos[0]);
    return item;
  }

  async function verwerkJob(job) {
    try {
      if (location.pathname.startsWith('/item/')) {
        job.results = job.results || [];
        job.results.push(await scrapeItemPage());
        sessionStorage.setItem('rep_job', JSON.stringify(job));
        const queue = JSON.parse(sessionStorage.getItem('rep_queue') || '[]');
        const next = queue.shift();
        sessionStorage.setItem('rep_queue', JSON.stringify(queue));
        if (next) {
          location.href = 'https://doppel.fit' + next;
        } else {
          await postResult({ id: job.id, items: job.results });
          sessionStorage.removeItem('rep_job');
          sessionStorage.removeItem('rep_queue');
        }
        return;
      }
      if (location.pathname === '/s') {
        await sleep(3500); // resultaten laden
        const hrefs = [...new Set(Array.from(document.querySelectorAll('a[href^="/item/"]'))
          .map((a) => a.getAttribute('href'))
          .filter((h) => h && h.length > 5))];
        const kies = hrefs.slice(0, Math.max(1, job.count || 3));
        sessionStorage.setItem('rep_queue', JSON.stringify(kies.slice(1)));
        if (kies.length) {
          location.href = 'https://doppel.fit' + kies[0];
        } else {
          await postResult({ id: job.id, items: [], error: 'geen resultaten' });
          sessionStorage.removeItem('rep_job');
        }
        return;
      }
      location.href = job.url; // onbekende pagina: ga naar de zoek-URL
    } catch (e) {
      await postResult({ id: job.id, error: String(e).slice(0, 200) });
      sessionStorage.removeItem('rep_job');
      sessionStorage.removeItem('rep_queue');
    }
  }

  async function poll() {
    let job = null;
    try {
      const raw = sessionStorage.getItem('rep_job');
      job = raw ? JSON.parse(raw) : null;
      if (!job) {
        const r = await fetch(BACK + '/whatsapp/source/next', { headers: HDRS });
        if (r.ok) {
          const d = await r.json();
          if (d && d.job) {
            job = d.job;
            job.results = [];
            sessionStorage.setItem('rep_job', JSON.stringify(job));
            sessionStorage.setItem('rep_queue', '[]');
            location.href = job.url;
            return;
          }
        }
        return;
      }
    } catch (e) { return; }
    await verwerkJob(job);
  }

  setInterval(poll, 7000);
  setTimeout(poll, 3000);
})();
