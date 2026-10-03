/**
 * Rep Agent — WhatsApp bridge (Baileys, gekoppeld apparaat).
 *
 * Koppelt je BESTAANDE WhatsApp-nummer aan de bot, net zoals WhatsApp Web:
 * - je telefoon blijft gewoon werken, alle chats blijven zichtbaar;
 * - typ jij zelf iets in een chat (vanaf je telefoon), dan pauzeert de bot
 *   voor die chat ("human takeover", standaard 12u, instelbaar via WA_TAKEOVER_HOURS);
 *   "/bot uit" = 7 dagen pauze, "/bot aan" = bot weer aan;
 * - inkomende tekst/vocenotes/berichten-met-onderschrift gaan naar de Python-backend
 *   (POST /whatsapp/incoming) en het antwoord gaat terug naar de chat.
 *
 * Env vars:
 *   BACKEND_URL  default http://127.0.0.1:7860
 *   BACKEND_KEY  X-Access-Code voor de backend (ACCESS_CODE), optioneel
 *   WA_SESSION_DIR  default ../data/whatsapp-session (Baileys-sessie)
 *   WA_ALLOW     kommalijst nummers die de bot WEL bedient (leeg = iedereen 1-op-1)
 *   WA_BLOCK     kommalijst nummers die de bot NOOIT bedient
 */
const path = require('path');
const fs = require('fs');

const baileys = require('@whiskeysockets/baileys');
const makeWASocket = baileys.default;
const {
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
  downloadMediaMessage,
} = baileys;
const pino = require('pino');
const qrcodeLib = require('qrcode');
const qrcodeTerminal = require('qrcode-terminal');

const BACKEND_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:7860').replace(/\/+$/, '');
const BACKEND_KEY = process.env.BACKEND_KEY || '';
const SESSION_DIR = process.env.WA_SESSION_DIR ||
  path.join(__dirname, '..', 'data', 'whatsapp-session');
const WA_ALLOW = (process.env.WA_ALLOW || '').split(',').map((s) => s.trim()).filter(Boolean);
const WA_BLOCK = (process.env.WA_BLOCK || '').split(',').map((s) => s.trim()).filter(Boolean);
const WA_PAIR_PHONE = (process.env.WA_PAIR_PHONE || '31684805378').replace(/[^0-9]/g, '');

const logger = pino({ level: 'silent' });

function log(...args) {
  console.log(new Date().toISOString(), ...args);
}

async function post(pathName, body, timeoutMs = 15000) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const r = await fetch(BACKEND_URL + pathName, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Access-Code': BACKEND_KEY },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (r.ok) return await r.json().catch(() => ({}));
      log('backend', pathName, 'HTTP', r.status);
    } catch (e) {
      log('backend', pathName, 'fout:', e.message);
    }
    await new Promise((res) => setTimeout(res, 1500 * attempt));
  }
  return {};
}

function jidNumber(jid) {
  return String(jid || '').split('@')[0].split(':')[0];
}

function messageText(m) {
  let inner = m.message || {};
  inner = inner.ephemeralMessage?.message || inner.viewOnceMessage?.message ||
    inner.documentWithCaptionMessage?.message || inner;
  if (inner.conversation) return { type: 'text', text: inner.conversation };
  if (inner.extendedTextMessage?.text) return { type: 'text', text: inner.extendedTextMessage.text };
  if (inner.imageMessage) return { type: 'image', text: inner.imageMessage.caption || '' };
  if (inner.videoMessage) return { type: 'video', text: inner.videoMessage.caption || '' };
  if (inner.audioMessage) return { type: 'audio', text: '' };
  if (inner.documentMessage?.caption) return { type: 'text', text: inner.documentMessage.caption };
  return null;
}

function chunks(text, max = 3500) {
  const out = [];
  let rest = String(text || '');
  while (rest.length > max) {
    let cut = rest.lastIndexOf('\n', max);
    if (cut < max * 0.5) cut = max;
    out.push(rest.slice(0, cut));
    rest = rest.slice(cut).replace(/^\n+/, '');
  }
  if (rest) out.push(rest);
  return out;
}

async function handleMessage(sock, m, seen, mySends) {
  if (!m.key || seen.has(m.key.id)) return;
  seen.add(m.key.id);
  if (seen.size > 800) seen.clear();

  const jid = m.key.remoteJid || '';
  if (!jid || jid === 'status@broadcast' || jid.endsWith('@g.us') || jid.endsWith('@newsletter')) return;

  // Eigen echo's van berichten die de bridge zelf stuurde -> helemaal negeren
  // (anders zou de bot zijn eigen antwoorden zien als "Younes reageerde zelf").
  if (m.key.fromMe && mySends.has(m.key.id)) return;

  const parsed = messageText(m);
  if (!parsed) return; // stickers, locaties, contacten e.d. -> geen actie

  const num = jidNumber(jid);
  if (!m.key.fromMe) {
    if (WA_BLOCK.some((b) => num.startsWith(b))) return;
    if (WA_ALLOW.length && !WA_ALLOW.some((a) => num.startsWith(a))) return;
  }

  const fromMe = !!m.key.fromMe;
  log(`${fromMe ? 'IK      ' : 'KLANT   '} +${num} (${m.pushName || 'onbekend'}):`,
    (parsed.text || parsed.type).slice(0, 90).replace(/\n/g, ' '));

  let audioB64 = null;
  if (parsed.type === 'audio') {
    try {
      const buf = await downloadMediaMessage(m, 'buffer', {
        reuploadRequest: sock.updateMediaMessage,
      });
      audioB64 = Buffer.from(buf).toString('base64');
    } catch (e) {
      log('audio downloaden mislukt:', e.message);
      return;
    }
  }

  try { await sock.readMessages([m.key]); } catch (_) { /* geen probleem */ }

  // Snelle bevestiging naar de klant zodra het echte antwoord even duurt
  // (zoeken/prijzen checken); korte vragen zijn al beantwoord vóór de timer.
  let ackTimer = null;
  if (!fromMe) {
    ackTimer = setTimeout(async () => {
      try {
        const s = await sock.sendMessage(jid, { text: 'Ik ga even voor je kijken 👍' });
        if (s && s.key && s.key.id) mySends.add(s.key.id);
      } catch (_) { /* geen ramp */ }
    }, 4500);
  }

  const resp = await post('/whatsapp/incoming', {
    chat: jid,
    num,
    name: m.pushName || '',
    from_me: fromMe,
    type: parsed.type,
    text: (parsed.text || '').trim(),
    audio_b64: audioB64,
  }, 150000);
  if (ackTimer) clearTimeout(ackTimer);

  const reply = resp && resp.reply;
  const toMe = resp && resp.to_me;
  const images = (resp && Array.isArray(resp.images) && resp.images) || [];
  if (reply && (!fromMe || toMe)) {
    try {
      await sock.sendPresenceUpdate('composing', jid);
      await new Promise((res) => setTimeout(res, 800 + Math.random() * 1400));
      let rest = reply;
      if (images.length) {
        // eerste foto direct meesturen (al gecropt door de doppel-bridge), rest als tekst
        const caption = chunks(rest, 900)[0] || '';
        try {
          const s1 = await sock.sendMessage(jid, { image: Buffer.from(images[0], 'base64'), caption });
          if (s1 && s1.key && s1.key.id) mySends.add(s1.key.id);
          rest = rest.slice(caption.length).replace(/^\s+/, '');
          for (const extra of images.slice(1, 3)) {
            const s2 = await sock.sendMessage(jid, { image: Buffer.from(extra, 'base64') });
            if (s2 && s2.key && s2.key.id) mySends.add(s2.key.id);
          }
        } catch (e) {
          log('foto versturen mislukt:', e.message);
        }
      }
      for (const part of chunks(rest)) {
        const s3 = await sock.sendMessage(jid, { text: part });
        if (s3 && s3.key && s3.key.id) mySends.add(s3.key.id);
        await new Promise((res) => setTimeout(res, 350));
      }
      if (mySends.size > 600) mySends.clear();
      await sock.sendPresenceUpdate('paused', jid);
    } catch (e) {
      log('versturen mislukt:', e.message);
    }
  }
}

async function start(backoffMs = 3000) {
  fs.mkdirSync(SESSION_DIR, { recursive: true });
  let bestanden = [];
  try { bestanden = fs.readdirSync(SESSION_DIR); } catch (_) { }
  log('start met sessie-map', SESSION_DIR, '| bestanden:', bestanden.length,
    bestanden.includes('creds.json') ? '(creds.json AANWEZIG)' : '(geen creds.json!)');
  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
  log('authState: registered =', !!state.creds?.registered,
    '| account =', (state.creds?.me?.id || state.creds?.account || 'onbekend').toString().slice(0, 20));
  const { version } = await fetchLatestBaileysVersion().catch(() => ({ version: undefined }));

  const sock = makeWASocket({
    version,
    auth: { creds: state.creds, keys: makeCacheableSignalKeyStore(state.keys, logger) },
    logger,
    printQRInTerminal: false,
    browser: ['Ubuntu', 'Chrome', '20.0.04'],
    markOnlineOnConnect: false,
    syncFullHistory: false,
  });

  const seen = new Set();
  const mySends = new Set(); // bericht-id's die DEZE bridge zelf verstuurde (echo's negeren)
  let nextBackoff = backoffMs;
  let pairRequested = false;
  let openedEver = false;

  // hang-guard: als 'verbinden' te lang duurt zonder QR of open, socket opnieuw
  const hangGuard = setTimeout(() => {
    if (!openedEver) {
      clearTimeout(hangGuard);
      log('verbinden blijft hangen — socket opnieuw starten');
      try { sock.end(new Error('hang-guard: connectie-timeout')); } catch (_) { }
      setTimeout(() => start(nextBackoff), 5000);
    }
  }, 80000);
  sock.ev.on('connection.update', (u2) => {
    if (u2.connection === 'open') { openedEver = true; clearTimeout(hangGuard); }
    if (u2.connection === 'close') clearTimeout(hangGuard);
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', async (u) => {
    const { connection, lastDisconnect, qr } = u;
    if (qr && WA_PAIR_PHONE && !pairRequested && !sock.authState?.creds?.registered) {
      // Koppelen MET CODE (handig als je alleen je telefoon hebt): geen QR nodig.
      pairRequested = true;
      try {
        await new Promise((res) => setTimeout(res, 2500));
        const raw = await sock.requestPairingCode(WA_PAIR_PHONE);
        const code = String(raw || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
        const pretty = code.length === 8 ? `${code.slice(0, 4)}-${code.slice(4)}` : code;
        await post('/whatsapp/qr', { pair_code: pretty });
        log('koppelcode gepost voor', WA_PAIR_PHONE.slice(0, 4), '***');
      } catch (e) {
        log('koppelcode opvragen mislukt:', e.message);
        pairRequested = false;
      }
    }
    if (qr) {
      const dataUrl = await qrcodeLib.toDataURL(qr).catch(() => null);
      await post('/whatsapp/qr', { qr: dataUrl || qr, qr_raw: qr });
      log('nieuwe QR-code gepost naar backend — scan met je telefoon of gebruik de koppelcode');
    }
    if (connection === 'open') {
      log('WhatsApp VERBONDEN ✓ (telefoon blijft gewoon werken)');
      nextBackoff = 3000;
      await post('/whatsapp/status', { status: 'connected' });
    }
    if (connection === 'connecting') {
      await post('/whatsapp/status', { status: 'verbinden' });
    }
    if (connection === 'close') {
      const code = lastDisconnect?.error?.output?.statusCode;
      const loggedOut = code === DisconnectReason.loggedOut;
      log('verbinding gesloten, code:', code, '| loggedOut:', loggedOut);
      await post('/whatsapp/status', {
        status: loggedOut ? 'uitgelogd' : 'herverbinden',
        error: String((lastDisconnect && lastDisconnect.error) || ''),
      });
      if (loggedOut) {
        log('sessie ongeldig — oude sessie ook uit HF-backup laten wissen, dan map leeg');
        await post('/whatsapp/logout', {});
        fs.rmSync(SESSION_DIR, { recursive: true, force: true });
      }
      setTimeout(() => start(nextBackoff).catch((e2) => log('herstart mislukt:', e2.message)), nextBackoff);
      nextBackoff = Math.min(nextBackoff * 2, 60000);
    }
  });

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;
    for (const m of messages || []) {
      try {
        await handleMessage(sock, m, seen, mySends);
      } catch (e) {
        log('berichtverwerking fout:', e.message);
      }
    }
  });
}

log('Rep Agent WhatsApp-bridge start | backend:', BACKEND_URL, '| sessie:', SESSION_DIR);
if (!BACKEND_KEY) log('LET OP: BACKEND_KEY niet gezet (nodig als de backend ACCESS_CODE gebruikt)');

// never-die: crashes mogen het proces niet killen (supervisor in start.sh vangt de rest)
process.on('uncaughtException', (e) => log('uncaughtException (blijf draaien):', e.message));
process.on('unhandledRejection', (e) => log('unhandledRejection (blijf draaien):', String(e)));

start().catch((e) => {
  log('start mislukt:', e.message, '- over 10s opnieuw');
  setTimeout(() => start().catch((e2) => log('herstart mislukt:', e2.message)), 10000);
});
