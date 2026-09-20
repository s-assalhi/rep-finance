"""
Rep Finance — Telegram voice bot + web dashboard in one process.
Designed for a free Hugging Face Docker Space (CPU).

Flow:  Telegram voice (NL) -> faster-whisper ASR -> GLM (OpenAI-compatible)
       -> JSON action -> ledger (data/ledger.json, optional HF Dataset sync)
       -> Dutch confirmation reply + live dashboard on "/".

Env vars (set as Space secrets):
  TELEGRAM_TOKEN   required  - from @BotFather
  LLM_API_KEY      required  - Z.ai / OpenAI-compatible key
  LLM_BASE_URL     optional  - default https://api.z.ai/api/coding/paas/v4
  LLM_MODEL        optional  - default glm-4.6
  WHISPER_MODEL    optional  - tiny | base (default) | small
  LEDGER_PATH      optional  - default data/ledger.json
  HF_DATASET       optional  - "username/rep-ledger" private dataset for backup
  HF_TOKEN         optional  - HF write token, needed only for HF_DATASET sync
"""
import asyncio
import json
import os
import re
import threading
import time

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------- config ----------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-4.6")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
LEDGER_PATH = os.environ.get("LEDGER_PATH", "data/ledger.json")
HF_DATASET = os.environ.get("HF_DATASET", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()  # dashboard-gate op publieke Space
BASETAO_COOKIE = os.environ.get("BASETAO_COOKIE", "").strip()  # DevTools cookie voor /basetao sync

ORDER_STATUSES = ["interesse", "info_gevraagd", "prijs_gegeven", "wacht_op_antwoord",
                  "te_bestellen", "besteld", "onderweg", "binnen", "verpakken", "klaar",
                  "geen_interesse"]
PAYMENT_STATUSES = ["nog_niet_betaald", "deels_betaald", "betaald", "geen_interesse"]

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def order_add(d, customer, items, price_eur, payment_method="onbekend", basetao_ids=None, note=""):
    orders = d.setdefault("orders", [])
    nxt = 1 + max([o.get("num", 0) for o in orders] or [0])
    o = {"num": nxt, "customer": customer or "?", "items": items or "",
         "price_eur": float(price_eur or 0), "order_status": "te_bestellen",
         "payment_status": "nog_niet_betaald", "payment_method": payment_method,
         "basetao_ids": basetao_ids or [], "created": _now(), "updated": _now(),
         "history": [{"ts": _now(), "event": "order aangemaakt"}]}
    if note:
        o["note"] = note
    orders.append(o)
    return o

def order_update(d, num, **fields):
    for o in d.get("orders", []):
        if o.get("num") == num:
            for k, v in fields.items():
                if v in (None, ""):
                    continue
                if k in ("order_status", "payment_status") and v not in (ORDER_STATUSES + PAYMENT_STATUSES):
                    continue
                if o.get(k) != v:
                    o.setdefault("history", []).append(
                        {"ts": _now(), "field": k, "from": o.get(k), "to": v})
                    o[k] = v
            o["updated"] = _now()
            return o
    return None

API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
_ledlock = threading.Lock()

# ---------------- ledger ----------------
def ledger_load():
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def ledger_save(d):
    os.makedirs(os.path.dirname(LEDGER_PATH) or ".", exist_ok=True)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def hf_sync_up():
    """Mirror the ledger to a private HF Dataset so restarts lose nothing."""
    if not (HF_DATASET and HF_TOKEN):
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=LEDGER_PATH, path_in_repo="ledger.json",
            repo_id=HF_DATASET, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print("hf sync up failed:", e)

def hf_sync_down():
    """Bij start: de HF-backup is leidend als die bestaat (repo-seed is alleen fallback)."""
    if not (HF_DATASET and HF_TOKEN):
        return
    try:
        import shutil
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename="ledger.json", token=HF_TOKEN)
        os.makedirs(os.path.dirname(LEDGER_PATH) or ".", exist_ok=True)
        shutil.copy(p, LEDGER_PATH)
        print("ledger hersteld uit HF dataset")
    except Exception as e:  # noqa: BLE001
        print("hf sync down failed:", e)

def compute_stats(d):
    inc_cash = inc_wallet = cost = 0.0
    for e in d["entries"]:
        try:
            a = float(e.get("amount_eur") or 0)
        except (TypeError, ValueError):
            continue
        if e.get("type") == "income":
            if e.get("method") == "cash":
                inc_cash += a
            else:
                inc_wallet += a
        elif e.get("type") == "cost":
            cost += a
    income = inc_cash + inc_wallet
    profit = income - cost
    margin = (profit / income * 100.0) if income else 0.0
    cash_pct = (inc_cash / income * 100.0) if income else 0.0
    open_orders = [o for o in d.get("orders", []) if o.get("payment_status") != "betaald"]
    te_innen = sum(float(o.get("price_eur") or 0) for o in open_orders)
    return dict(
        inc_cash=round(inc_cash, 2), inc_wallet=round(inc_wallet, 2),
        income=round(income, 2), cost=round(cost, 2), profit=round(profit, 2),
        margin_pct=round(margin, 1), cash_pct=round(cash_pct, 1),
        wallet_pct=round(100 - cash_pct, 1),
        gap_pct=round(cash_pct - margin, 1),
        open_orders=len(open_orders), te_innen=round(te_innen, 2))

def stats_text_nl():
    s = compute_stats(ledger_load())
    return (f"📊 Cash €{s['inc_cash']:g} ({s['cash_pct']}%) | "
            f"Basetao €{s['inc_wallet']:g} ({s['wallet_pct']}%)\n"
            f"💰 Inkomsten €{s['income']:g} − kosten €{s['cost']:g} = "
            f"€{s['profit']:g} (marge {s['margin_pct']}%)\n"
            f"🎯 Cash% − marge% verschil: {s['gap_pct']} pct-punt"
            + (" ✅ in balans" if abs(s['gap_pct']) < 5 else ""))

def apply_action(a, raw):
    t = a.get("type", "note")
    if t == "query":
        return stats_text_nl()
    if t == "order":
        with _ledlock:
            d = ledger_load()
            o = order_add(d, a.get("customer"), a.get("items"), a.get("amount_eur") or 0,
                          note=raw)
            ledger_save(d)
            hf_sync_up()
        return (f"📝 Order #{o['num']} aangemaakt: {o['customer']} — {o['items'] or '?'} "
                f"€{o['price_eur']:g}\n📦 status: te bestellen | 💶 nog niet betaald\n"
                f"Wijzig met: /status #{o['num']} besteld  of  /betaald #{o['num']} cash")
    amt = a.get("amount_eur")
    with _ledlock:
        d = ledger_load()
        if t in ("income", "cost") and amt:
            entry = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": t, "amount_eur": float(amt),
                "method": a.get("method") or ("cash" if t == "income" else "basetao"),
                "customer": a.get("customer"), "items": a.get("items"),
                "note": a.get("note") or raw, "source": "telegram",
            }
            d["entries"].append(entry)
            ledger_save(d)
            s = compute_stats(d)
            hf_sync_up()
            if t == "income":
                meth = "cash" if entry["method"] == "cash" else "basetao-portemonnee"
                who = f" van {entry['customer']}" if entry.get("customer") else ""
                return (f"✅ Inkomsten bijgeschreven: €{amt:g} ({meth}){who}\n"
                        f"📊 Cash {s['cash_pct']}% | Basetao {s['wallet_pct']}% | "
                        f"Marge {s['margin_pct']}%")
            return (f"✅ Kosten geboekt: €{amt:g}\n"
                    f"📊 Inkomsten €{s['income']:g} vs kosten €{s['cost']:g} → "
                    f"marge {s['margin_pct']}%")
        d["entries"].append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "type": "note", "note": raw, "source": "telegram"})
        ledger_save(d)
        hf_sync_up()
        return "📝 Notitie opgeslagen."

# ---------------- ASR (faster-whisper, local CPU) ----------------
_whisper = None

def transcribe(path):
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("loading whisper model:", WHISPER_MODEL)
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segs, _ = _whisper.transcribe(path, language="nl", beam_size=1)
    return " ".join(s.text.strip() for s in segs).strip()

# ---------------- LLM (OpenAI-compatible) ----------------
SCHEMA_PROMPT = """Je zet Nederlandse berichten om naar bookhoud-acties. Antwoord ONLY met JSON, geen andere tekst:
{"type":"income|cost|order|query|note","amount_eur":number|null,"method":"cash|basetao"|null,"customer":string|null,"items":string|null,"note":string|null}
Regels:
- "cash" = contant geld (physical euro cash). "basetao" = directe betaling in de basetao-portemonnee.
- amount_eur altijd in euro's (converteer "lek"/"bale"/"lak" naar eur getal).
- type=order als iemand iets bestelt of besteld heeft (klant + items + bedrag) maar er nog geen geld ontvangen is.
- type=query als er om totalen/overzicht gevraagd wordt; type=note als het geen inkomsten/kosten/bestelling/vraag is.
- customer = wie betaalt/gaf opdracht; items = wat is er gekocht/besteld."""

def llm_parse(text):
    if not LLM_API_KEY:
        return {"type": "note", "note": text}
    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={"model": LLM_MODEL, "temperature": 0,
              "messages": [{"role": "system", "content": SCHEMA_PROMPT},
                           {"role": "user", "content": text}]},
        timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0) if m else content)

EXTRACT_PROMPT = """Haal uit deze Basetao-orderlijst alle producten. Antwoord ONLY met een JSON-array, geen andere tekst:
[{"order_id":"1936771","title":"productnaam","price_cny":123.45}]
- order_id = het Basetao-ordernummer bij het product (alleen cijfers)
- title = de product/mdl-omschrijving
- price_cny = de prijs in CNY-nummer (null als niet vermeld)
Sla dubbelloopse kopregels/paginering over. Geen producten? antwoord []"""

def llm_extract_products(text):
    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={"model": LLM_MODEL, "temperature": 0,
              "messages": [{"role": "system", "content": EXTRACT_PROMPT},
                           {"role": "user", "content": text[:14000]}]},
        timeout=150)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", content, re.S)
    return json.loads(m.group(0) if m else "[]")

# ---------------- telegram ----------------
def tg(method, **kw):
    try:
        requests.post(f"{API}/{method}", json=kw, timeout=30)
    except Exception as e:  # noqa: BLE001
        print("tg error:", method, e)

def tg_download(file_id):
    try:
        r = requests.get(f"{API}/getFile", params={"file_id": file_id}, timeout=30).json()
        path = r["result"]["file_path"]
        local = os.path.join("/tmp", f"voice_{int(time.time())}.ogg")
        with open(local, "wb") as f:
            f.write(requests.get(f"{API}/file/{path}", timeout=120).content)
        return local
    except Exception as e:  # noqa: BLE001
        print("download error:", e)
        return None

def basetao_snapshot():
    """Basetao-sessie testen + wat er server-side leesbaar is (raw HTML)."""
    headers = {
        "Cookie": BASETAO_COOKIE,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
        "Referer": "https://www.basetao.com/",
        "Accept": "text/html",
    }
    r = requests.get(
        "https://www.basetao.com/best-taobao-agent-service/my_account/welcome.html",
        headers=headers, timeout=30)
    t = r.text
    logged_in = "Welcome back" in t and "Login" not in t[:3000]
    m = re.search(r'bi-currency-yen[\s\S]{0,150}?>\s*([\d.,]+)\s*<', t)
    counters = dict(re.findall(
        r'id="(Ordered|Arrived|Cancelled|Shipped|Searching|Received|Pending)"'
        r'[\s\S]{0,300}?badge[^>]*>\s*(\d+)\s*</span>', t))
    return {"logged_in": logged_in, "http": r.status_code,
            "balance_cny": m.group(1) if m else None,
            "counters": counters,
            "note": "tellers/saldo worden door basetao via JS geladen; "
                    "gebruik de browser-bridge voor volledige data"}

def handle_update(msg):
    chat_id = msg["chat"]["id"]
    media = msg.get("voice") or msg.get("audio") or msg.get("document")
    if media:
        tg("sendChatAction", chat_id=chat_id, action="typing")
        path = tg_download(media["file_id"])
        if not path:
            tg("sendMessage", chat_id=chat_id, text="❌ Kon voicebestand niet ophalen.")
            return
        try:
            text = transcribe(path)
        except Exception as e:  # noqa: BLE001
            print("asr error:", e)
            tg("sendMessage", chat_id=chat_id, text="❌ Spraakherkenning mislukt.")
            return
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if not text:
            tg("sendMessage", chat_id=chat_id, text="❓ Geen spraak herkend.")
            return
        tg("sendMessage", chat_id=chat_id, text=f"🎙️ \"{text}\"")
    elif msg.get("text"):
        text = msg["text"].strip()
    else:
        return
    low = text.lower()
    if low.startswith("/basetao") or low.startswith("/orders") or low.startswith("/order ") \
            or low.startswith("/status ") or low.startswith("/betaald") or low.startswith("/klant "):
        return handle_command(chat_id, text)
    try:
        action = llm_parse(text)
    except Exception as e:  # noqa: BLE001
        print("llm error:", e)
        tg("sendMessage", chat_id=chat_id, text="❌ Kon de opdracht niet verwerken.")
        return
    tg("sendMessage", chat_id=chat_id, text=apply_action(action, text))

def handle_command(chat_id, text):
    low = text.lower().strip()
    if low.startswith("/basetao"):
        if not BASETAO_COOKIE:
            tg("sendMessage", chat_id=chat_id,
               text="❌ BASETAO_COOKIE is niet ingesteld (Render secret).")
            return
        tg("sendChatAction", chat_id=chat_id, action="typing")
        snap = basetao_snapshot()
        if not snap or not snap.get("logged_in"):
            tg("sendMessage", chat_id=chat_id,
               text="⚠️ Basetao-cookie verlopen of geblokkeerd — ververs hem (Copy as cURL → mij sturen).")
            return
        c = snap.get("counters", {})
        tg("sendMessage", chat_id=chat_id,
           text=(f"📦 Basetao — saldo ¥{snap['balance_cny']}\n"
                 f"Ordered {c.get('Ordered', '?')} | Arrived {c.get('Arrived', '?')} | "
                 f"Shipped {c.get('Shipped', '?')} | Searching {c.get('Searching', '?')}\n"
                 f"Pakketten ontvangen: {c.get('Received', '?')}"))
        return
    with _ledlock:
        d = ledger_load()
        orders = d.get("orders", [])
        if low.startswith("/order "):
            parts = [p.strip() for p in text[6:].split("|")]
            customer = parts[0] if parts else "?"
            items = parts[1] if len(parts) > 1 else ""
            price = parts[2] if len(parts) > 2 else "0"
            o = order_add(d, customer, items, price)
            ledger_save(d)
            hf_sync_up()
            tg("sendMessage", chat_id=chat_id,
               text=(f"📝 Order #{o['num']}: {o['customer']} — {o['items'] or '?'} €{o['price_eur']:g}\n"
                     f"📦 te bestellen | 💶 nog niet betaald"))
            return
        if low.startswith("/orders"):
            if not orders:
                tg("sendMessage", chat_id=chat_id, text="📭 Nog geen orders.")
                return
            lines = []
            for o in orders[-15:]:
                lines.append(f"#{o['num']} {o['customer']} — {o['items'] or '?'} €{o['price_eur']:g} — "
                             f"📦{o['order_status']} 💶{o['payment_status']}")
            tg("sendMessage", chat_id=chat_id, text="📋 Orders:\n" + "\n".join(lines))
            return
        m = re.match(r"/status\s+#?(\d+)\s+(\S+)", low)
        if m:
            num, st = int(m.group(1)), m.group(2)
            if st not in ORDER_STATUSES:
                tg("sendMessage", chat_id=chat_id,
                   text="⚠️ Onbekende status. Kies uit: " + ", ".join(ORDER_STATUSES))
                return
            o = order_update(d, num, order_status=st)
            ledger_save(d)
            hf_sync_up()
            tg("sendMessage", chat_id=chat_id,
               text=(f"📦 Order #{num}: status → {st}" if o else f"❌ Order #{num} niet gevonden."))
            return
        m = re.match(r"/betaald\s+#?(\d+)(?:\s+(cash|basetao))?", low)
        if m:
            num, meth = int(m.group(1)), m.group(2)
            fields = {"payment_status": "betaald"}
            if meth:
                fields["payment_method"] = meth
            o = order_update(d, num, **fields)
            ledger_save(d)
            hf_sync_up()
            tg("sendMessage", chat_id=chat_id,
               text=(f"💶 Order #{num}: betaald" + (f" via {meth}" if meth else "") + " ✅"
                     if o else f"❌ Order #{num} niet gevonden."))
            return
        m = re.match(r"/klant\s+#?(\d+)\s+(.+)", low)
        if m:
            num, naam = int(m.group(1)), text.split(None, 2)[2].strip()
            o = order_update(d, num, customer=naam)
            ledger_save(d)
            hf_sync_up()
            tg("sendMessage", chat_id=chat_id,
               text=(f"👤 Order #{num}: klant → {naam}" if o else f"❌ Order #{num} niet gevonden."))
            return
        tg("sendMessage", chat_id=chat_id,
           text="ℹ️ Gebruik: /order klant | items | bedrag · /orders · /status #1 besteld · /betaald #1 cash · /klant #1 Koppig")

def poll_loop():
    offset = 0
    print("telegram polling started, token set:", bool(TELEGRAM_TOKEN))
    while True:
        if not TELEGRAM_TOKEN:
            time.sleep(30)
            continue
        try:
            r = requests.get(
                f"{API}/getUpdates",
                params={"timeout": 25, "offset": offset,
                        "allowed_updates": json.dumps(["message"])},
                timeout=35)
            for u in r.json().get("result", []):
                offset = u["update_id"] + 1
                try:
                    handle_update(u.get("message") or {})
                except Exception as e:  # noqa: BLE001
                    print("handle error:", e)
        except Exception as e:  # noqa: BLE001
            print("poll error:", e)
            time.sleep(5)

# ---------------- web dashboard ----------------
DASH = """<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>Rep Finance</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0f1716;color:#e8efec;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px}.sub{color:#8aa39c;font-size:13px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;max-width:980px}
.card{background:#182522;border:1px solid #24382f;border-radius:12px;padding:16px}
.k{color:#8aa39c;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.v{font-size:26px;font-weight:700;margin-top:6px}
.bar{height:14px;border-radius:7px;overflow:hidden;display:flex;margin-top:10px;background:#0c1210}
.bar span{height:100%}.cash{background:#4caf7d}.wallet{background:#3d7dd8}
.gap-ok{color:#4caf7d}.gap-bad{color:#e0a13d}
table{width:100%;max-width:980px;border-collapse:collapse;margin-top:22px;font-size:13px}
td,th{padding:7px 9px;border-bottom:1px solid #1e2f28;text-align:left}
th{color:#8aa39c;font-weight:600}form{margin-top:26px;max-width:980px;background:#182522;
border:1px solid #24382f;border-radius:12px;padding:16px;display:grid;
grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
input,select{background:#0c1210;border:1px solid #2b463c;color:#e8efec;border-radius:8px;padding:8px;width:100%}
button{background:#4caf7d;border:0;border-radius:8px;padding:10px;font-weight:700;cursor:pointer}
.note{color:#8aa39c;font-size:12px;margin-top:18px;max-width:980px}
</style></head><body>
<h1>💶 Rep Finance — cash vs basetao</h1>
<div class="sub">live · verversen elke 60s · stuur stemberichten naar je Telegram bot</div>
<div class="grid">
<div class="card"><div class="k">Cash</div><div class="v">€__CASH__</div>
<div class="bar"><span class="cash" style="width:__CASH_PCT__%"></span><span class="wallet" style="width:__WALLET_PCT__%"></span></div>
<div class="k" style="margin-top:6px">__CASH_PCT__% van inkomsten</div></div>
<div class="card"><div class="k">Basetao portemonnee</div><div class="v">€__WALLET__</div>
<div class="k" style="margin-top:6px">__WALLET_PCT__% van inkomsten</div></div>
<div class="card"><div class="k">Winstmarge</div><div class="v">__MARGIN__%</div>
<div class="k" style="margin-top:6px">€__PROFIT__ winst op €__INCOME__</div></div>
<div class="card"><div class="k">Cash% vs marge%</div><div class="v __GAPCLS__">__GAP__ pp</div>
<div class="k" style="margin-top:6px">doel: ~0 (cash aandeel volgt winst)</div></div>
<div class="card"><div class="k">Open orders</div><div class="v">__OPENORDERS__</div>
<div class="k" style="margin-top:6px">te innen: €__TEINNEN__</div></div>
<div class="card"><div class="k">Basetao portemonnee</div><div class="v">¥__BTBAL__</div>
<div class="k" style="margin-top:6px">__BTCNT__</div></div>
</div>
<h1 style="font-size:17px;margin-top:30px">📋 Order- &amp; betaalstatus per order</h1>
<table><tr><th>#</th><th>Klant</th><th>Items</th><th>€</th><th>Orderstatus</th><th>Betaalstatus</th><th>Basetao</th><th>Laatst</th><th></th></tr>
__OROWS__
</table>
<form onsubmit="addOrder(event)">
<input id="o_customer" placeholder="klant">
<input id="o_items" placeholder="items (bv. Ajax setje M)">
<input id="o_price" type="number" step="0.01" placeholder="prijs €">
<button>Nieuwe order</button></form>
<form onsubmit="updOrder(event)">
<input id="u_num" type="number" placeholder="order #">
<input id="u_customer" placeholder="klant (optioneel)">
<select id="u_os"><option value="">— orderstatus —</option>__OSOPT__</select>
<select id="u_ps"><option value="">— betaalstatus —</option>__PSOPT__</select>
<button>Status bijwerken</button></form>
<table><tr><th>Datum</th><th>Type</th><th>€</th><th>Methode</th><th>Klant</th><th>Notitie</th></tr>
__ROWS__
</table>
<form onsubmit="add(event)">
<input id="f_amount" type="number" step="0.01" placeholder="bedrag €">
<select id="f_type"><option value="income">inkomsten</option><option value="cost">kosten</option></select>
<select id="f_method"><option value="cash">cash</option><option value="basetao">basetao</option></select>
<input id="f_customer" placeholder="klant">
<input id="f_note" placeholder="notitie">
<button>Toevoegen</button></form>
<script>const K=new URLSearchParams(location.search).get('key')||(document.cookie.split('; ').find(r=>r.startsWith('key='))||'').slice(4)||'';
if(K)document.cookie='key='+K+';path=/;max-age=31536000';
async function add(e){e.preventDefault();const g=i=>document.getElementById(i).value;
const r=await fetch('/api/entry',{method:'POST',headers:{'Content-Type':'application/json','X-Access-Code':K},
body:JSON.stringify({type:g('f_type'),amount_eur:parseFloat(g('f_amount')),
method:g('f_method'),customer:g('f_customer')||null,note:g('f_note')||null})});
r.ok?location.reload():alert('mislukt');}
async function postOrder(b){const r=await fetch('/api/order',{method:'POST',
headers:{'Content-Type':'application/json','X-Access-Code':K},body:JSON.stringify(b)});
r.ok?location.reload():alert('mislukt');}
async function addOrder(e){e.preventDefault();const g=i=>document.getElementById(i).value;
postOrder({customer:g('o_customer'),items:g('o_items'),price_eur:parseFloat(g('o_price')||'0')});}
async function updOrder(e){e.preventDefault();const g=i=>document.getElementById(i).value;
const b={num:parseInt(g('u_num'))};if(g('u_os'))b.order_status=g('u_os');if(g('u_ps'))b.payment_status=g('u_ps');
if(g('u_customer'))b.customer=g('u_customer');
if(!g('u_num')||(!g('u_os')&&!g('u_ps')&&!g('u_customer'))){alert('vul order # en minstens één veld in');return;}
postOrder(b);}
async function delOrder(n){if(!confirm('Order #'+n+' verwijderen?'))return;
const r=await fetch('/api/order/'+n,{method:'DELETE',headers:{'X-Access-Code':K}});
r.ok?location.reload():alert('mislukt');}</script>
<div class="note">Kosten tot nu toe: €__COST__ · seed-data uit Rep_Database.xlsx (Codex-historie).</div>
</body></html>"""

def render_dashboard():
    w = basetao_wallet() or {}
    btbal = w.get("balance_cny") or "—"
    c = w.get("counters") or {}
    btc = ("ordered {0} · arrived {1} · shipped {2} · searching {3}".format(
        c.get("Pending", "?"), c.get("Arrived", "?"), c.get("Shipped", "?"),
        c.get("Searching", "?"))) if c else "live via basetao-API (cache 5 min)"
    with _ledlock:
        d = ledger_load()
        s = compute_stats(d)
        rows = []
        for e in reversed(d["entries"][-25:]):
            rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    e.get("ts", ""), e.get("type", ""),
                    ("€%g" % e["amount_eur"]) if e.get("amount_eur") else "—",
                    e.get("method") or "—", e.get("customer") or "—",
                    (e.get("note") or "")[:80]))
        orows = []
        for o in reversed(d.get("orders", [])):
            bt = " ".join('<a href="https://www.basetao.com/best-taobao-agent-service/'
                          'purchase/order_img/{0}.html" target="_blank" rel="noopener">{0}</a>'
                          .format(i) for i in o.get("basetao_ids", [])) or "—"
            orows.append(
                "<tr><td>#{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}{}</td><td>{}</td><td>{}</td>"
                "<td><button onclick=\"delOrder({})\" style=\"background:#7d3a3a;border:0;color:#fff;"
                "border-radius:6px;cursor:pointer;padding:2px 7px\">✖</button></td></tr>".format(
                    o.get("num", ""), o.get("customer", ""), o.get("items", ""),
                    ("€%g" % o["price_eur"]) if o.get("price_eur") else "—",
                    o.get("order_status", ""),
                    o.get("payment_status", ""),
                    (" (" + o["payment_method"] + ")") if o.get("payment_method") not in (None, "onbekend") else "",
                    bt, o.get("updated", ""), o.get("num", "")))
    os_opts = "".join(f'<option value="{s}">{s}</option>' for s in ORDER_STATUSES)
    ps_opts = "".join(f'<option value="{s}">{s}</option>' for s in PAYMENT_STATUSES)
    gap_cls = "gap-ok" if abs(s["gap_pct"]) < 5 else "gap-bad"
    return (DASH
            .replace("__CASH__", f"{s['inc_cash']:g}")
            .replace("__WALLET__", f"{s['inc_wallet']:g}")
            .replace("__CASH_PCT__", str(s["cash_pct"]))
            .replace("__WALLET_PCT__", str(s["wallet_pct"]))
            .replace("__MARGIN__", str(s["margin_pct"]))
            .replace("__PROFIT__", f"{s['profit']:g}")
            .replace("__INCOME__", f"{s['income']:g}")
            .replace("__COST__", f"{s['cost']:g}")
            .replace("__GAP__", str(s["gap_pct"]))
            .replace("__GAPCLS__", gap_cls)
            .replace("__OPENORDERS__", str(s["open_orders"]))
            .replace("__TEINNEN__", f"{s['te_innen']:g}")
            .replace("__BTBAL__", str(btbal))
            .replace("__BTCNT__", btc)
            .replace("__OROWS__", "\n".join(orows) or '<tr><td colspan="9">— nog geen orders —</td></tr>')
            .replace("__OSOPT__", os_opts)
            .replace("__PSOPT__", ps_opts)
            .replace("__ROWS__", "\n".join(rows) or "<tr><td colspan=6>—</td></tr>"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def access_gate(request: Request, call_next):
    if ACCESS_CODE and request.url.path != "/healthz":
        code = (request.query_params.get("key")
                or request.headers.get("x-access-code")
                or request.cookies.get("key"))
        if code != ACCESS_CODE:
            return JSONResponse({"error": "toegang geweigerd: voeg ?key=... toe"},
                                status_code=401)
    return await call_next(request)

@app.on_event("startup")
def _start():
    hf_sync_down()
    threading.Thread(target=poll_loop, daemon=True).start()

@app.get("/", response_class=HTMLResponse)
def index():
    return render_dashboard()

@app.get("/healthz")
def healthz():
    return {"ok": True, "bot_configured": bool(TELEGRAM_TOKEN)}

@app.get("/stats")
def stats():
    with _ledlock:
        d = ledger_load()
    return JSONResponse({"stats": compute_stats(d),
                         "recent": list(reversed(d["entries"][-25:]))})

@app.post("/api/entry")
async def api_entry(req: Request):
    b = await req.json()
    t, amt = b.get("type"), b.get("amount_eur")
    if t not in ("income", "cost") or not amt:
        return JSONResponse({"ok": False, "error": "type of bedrag ongeldig"}, status_code=400)
    with _ledlock:
        d = ledger_load()
        d["entries"].append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "type": t,
            "amount_eur": float(amt),
            "method": b.get("method") or ("cash" if t == "income" else "basetao"),
            "customer": b.get("customer"), "items": None,
            "note": b.get("note") or "website", "source": "website"})
        ledger_save(d)
        s = compute_stats(d)
        hf_sync_up()
    return JSONResponse({"ok": True, "stats": s})

@app.get("/orders")
def orders():
    with _ledlock:
        d = ledger_load()
    return JSONResponse({"orders": d.get("orders", [])})

_wallet_cache = {"ts": 0.0, "data": None}

def basetao_wallet(max_age=300):
    """Live basetao-saldo + tellers, max 1x per 5 min daadwerkelijk opgehaald."""
    if _wallet_cache["data"] and time.time() - _wallet_cache["ts"] < max_age:
        return _wallet_cache["data"]
    if BASETAO_COOKIE:
        try:
            _wallet_cache["data"] = basetao_snapshot()
            _wallet_cache["ts"] = time.time()
        except Exception as e:  # noqa: BLE001
            print("basetao wallet error:", e)
    return _wallet_cache["data"]

@app.post("/api/process-audio")
async def api_process_audio(req: Request):
    """Audio uploaden (ogg/mp3/wav) -> whisper -> GLM -> kasboek/orders."""
    body = await req.body()
    if len(body) < 500:
        return JSONResponse({"ok": False, "error": "geen audio ontvangen"}, status_code=400)
    path = os.path.join("/tmp", f"up_{int(time.time() * 1000)}.ogg")
    with open(path, "wb") as f:
        f.write(body)
    try:
        text = await asyncio.to_thread(transcribe, path)
    except Exception as e:  # noqa: BLE001
        print("asr error:", e)
        return JSONResponse({"ok": False, "error": "spraakherkenning mislukt"}, status_code=500)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if not text:
        return JSONResponse({"ok": False, "error": "geen spraak herkend"})
    try:
        action = await asyncio.to_thread(llm_parse, text)
    except Exception as e:  # noqa: BLE001
        print("llm error:", e)
        return JSONResponse({"ok": False, "transcript": text, "error": "verwerking mislukt"},
                            status_code=502)
    reply = apply_action(action, text)
    return JSONResponse({"ok": True, "transcript": text, "actie": action, "resultaat": reply})

@app.get("/basetao")
def basetao_route():
    if not BASETAO_COOKIE:
        return JSONResponse({"ok": False, "error": "BASETAO_COOKIE niet ingesteld"}, status_code=400)
    snap = basetao_snapshot()
    return JSONResponse({"ok": bool(snap and snap.get("logged_in")), **(snap or {})})

def basetao_fetch(path):
    headers = {
        "Cookie": BASETAO_COOKIE,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
        "Referer": "https://www.basetao.com/best-taobao-agent-service/my_account/welcome.html",
        "Accept": "text/html",
    }
    r = requests.get("https://www.basetao.com/best-taobao-agent-service/my_account/" + path,
                     headers=headers, timeout=40)
    return r.text if r.status_code == 200 else ""

@app.get("/basetao/rows")
def basetao_rows():
    """Debug: structuur van de orderlijst-pagina's zoals de server ze ziet."""
    out = {}
    for name, page in (("ordered", "order/ordered.html"), ("arrived", "order/arrived.html")):
        t = basetao_fetch(page)
        regions = []
        for m in list(re.finditer(r'order_img', t))[:4]:
            regions.append(t[max(0, m.start() - 700):m.start() + 400])
        out[name] = {"len": len(t), "has_login": "Login" in t[:3000],
                     "n_order_img": t.count("order_img"), "regions": regions}
    return JSONResponse(out)

@app.post("/api/basetao")
async def api_basetao(req: Request):
    """Browser-bridge: de ingelogde basetao-tab post hier haar dashboard-data."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    with _ledlock:
        d = ledger_load()
        d.setdefault("entries", []).append({
            "ts": _now(), "type": "note",
            "note": "basetao-sync: " + json.dumps(b, ensure_ascii=False)[:900],
            "source": "basetao-bridge"})
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True})

@app.post("/api/basetao/import")
async def api_basetao_import(req: Request):
    """Bridge: paginatext van basetao -> GLM -> orderregels (klant onbekend)."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    text = (b.get("text") or "").strip()
    page = b.get("page") or ("arrived" if "arrived" in (b.get("url") or "") else "ordered")
    if len(text) < 50:
        return JSONResponse({"ok": False, "error": "te weinig tekst"}, status_code=400)
    if not LLM_API_KEY:
        return JSONResponse({"ok": False, "error": "LLM_API_KEY ontbreekt"}, status_code=400)
    try:
        products = await asyncio.to_thread(llm_extract_products, text)
    except Exception as e:  # noqa: BLE001
        print("extract error:", e)
        return JSONResponse({"ok": False, "error": "extractie mislukt"}, status_code=502)
    status = "binnen" if page == "arrived" else "besteld"
    created, skipped = [], 0
    with _ledlock:
        d = ledger_load()
        existing = {str(bid) for o in d.get("orders", []) for bid in o.get("basetao_ids", [])}
        for p in products if isinstance(products, list) else []:
            if not isinstance(p, dict):
                continue
            bid = str(p.get("order_id") or "").strip()
            title = str(p.get("title") or "")[:120]
            if not title:
                continue
            if bid and bid in existing:
                skipped += 1
                continue
            o = order_add(d, "onbekend", title, 0, basetao_ids=[bid] if bid else [],
                          note=f"basetao {page}, kosten ¥{p.get('price_cny') or '?'}")
            order_update(d, o["num"], order_status=status)
            if bid:
                existing.add(bid)
            created.append(o["num"])
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True, "page": page, "aangemaakt": created,
                         "overslaan_duplicaat": skipped})

@app.post("/api/order")
async def api_order(req: Request):
    b = await req.json()
    with _ledlock:
        d = ledger_load()
        if b.get("num"):
            o = order_update(d, int(b["num"]), order_status=b.get("order_status"),
                             payment_status=b.get("payment_status"),
                             payment_method=b.get("payment_method"),
                             customer=b.get("customer"),
                             items=b.get("items"),
                             price_eur=b.get("price_eur"))
            if not o:
                return JSONResponse({"ok": False, "error": "order niet gevonden"}, status_code=404)
        else:
            if not b.get("customer"):
                return JSONResponse({"ok": False, "error": "klant ontbreekt"}, status_code=400)
            o = order_add(d, b.get("customer"), b.get("items"), b.get("price_eur") or 0,
                          payment_method=b.get("payment_method") or "onbekend",
                          basetao_ids=b.get("basetao_ids") or [], note="website")
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True, "order": {k: o.get(k) for k in
                        ("num", "customer", "items", "price_eur", "order_status", "payment_status")}})

@app.delete("/api/order/{num}")
async def api_order_delete(num: int):
    with _ledlock:
        d = ledger_load()
        before = len(d.get("orders", []))
        d["orders"] = [o for o in d.get("orders", []) if o.get("num") != num]
        if len(d["orders"]) == before:
            return JSONResponse({"ok": False, "error": "niet gevonden"}, status_code=404)
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True})
