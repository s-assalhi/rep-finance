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
import base64
from datetime import datetime
import io
import json
import mimetypes
import os
import re
import threading
import time

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

# ---------------- config ----------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-4.6")
LLM_THINKING = os.environ.get("LLM_THINKING", "disabled").strip().lower()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
LEDGER_PATH = os.environ.get("LEDGER_PATH", "data/ledger.json")
HF_DATASET = os.environ.get("HF_DATASET", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()  # dashboard-gate op publieke Space
BASETAO_COOKIE = os.environ.get("BASETAO_COOKIE", "").strip()  # DevTools cookie voor /basetao sync
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()  # gratis ASR: whisper-large-v3
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()  # Voxtral ASR (beste quadrant)
MISTRAL_ASR_MODEL = os.environ.get("MISTRAL_ASR_MODEL", "voxtral-small-latest")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()  # gratis ASR via AI Studio
WHATSAPP_ENABLED = os.environ.get("WHATSAPP_ENABLED", "0") == "1"  # Baileys-bridge start via start.sh
# Auto-pauze BOTSHARD uit (wens Younes, 03-10): de Render env-var staat (onveranderbaar
# via de API) op 12 en overrulet de code-default, dus hier bewust NIET meer uitlezen.
WA_TAKEOVER_HOURS = 0.0
WA_IMG_REPLY = os.environ.get(
    "WA_IMG_REPLY",
    "Ontvangen 👍 Zet er even tekst bij (wat zoek je, kleur/maat)? Dan pak ik het direct op.").strip()
WA_SESSION_DIR = os.environ.get("WA_SESSION_DIR", os.path.join("data", "whatsapp-session"))

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

# ---------------- voice-notities bewaren (altijd opnieuw transcribeerbaar) ----------------
_voice_ctx = {}  # chat_id -> file_id van de laatste stemnotitie in die chat

def voice_backup_up(audio_bytes, naam):
    """Stemnotitie meesturen naar de HF-dataset (audio/<naam>), zodat de audio
    nooit meer verloren gaat (de notities van 20-09 waren onherstelbaar weg)."""
    if not (HF_DATASET and HF_TOKEN):
        return
    try:
        from huggingface_hub import HfApi
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=io.BytesIO(audio_bytes),
            path_in_repo="audio/" + naam, repo_id=HF_DATASET, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print("voice backup failed:", e)

def voice_note_opslaan(text, source, chat_key=None, file_id=None, audio_bytes=None):
    """Transcript als note in het kasboek zetten + audio naar HF backuppen."""
    e = {"ts": _now(), "type": "note", "note": "🎙️ " + text[:1500], "source": source}
    if file_id:
        e["voice_file_id"] = file_id
    if chat_key:
        e["chat"] = chat_key
    with _ledlock:
        d = ledger_load()
        d.setdefault("entries", []).append(e)
        ledger_save(d)
    threading.Thread(target=hf_sync_up, daemon=True).start()
    if audio_bytes and HF_DATASET and HF_TOKEN:
        threading.Thread(target=voice_backup_up, daemon=True,
                         args=(audio_bytes, "voice_%d_%s.ogg" % (int(time.time()), source))).start()

def compute_stats(d):
    inc_cash = inc_bank = inc_wallet = cost = topup = 0.0
    for e in d["entries"]:
        try:
            a = float(e.get("amount_eur") or 0)
        except (TypeError, ValueError):
            continue
        t = e.get("type")
        if t == "income":
            m = e.get("method")
            if m == "cash":
                inc_cash += a
            elif m == "basetao":
                inc_wallet += a
            else:
                inc_bank += a  # bank/tikkie/ideal ontvangen
        elif t == "cost":
            cost += a
        elif t == "topup":
            topup += a
    income = inc_cash + inc_bank + inc_wallet
    profit = income - cost
    margin = (profit / income * 100.0) if income else 0.0
    cash_pct = (inc_cash / income * 100.0) if income else 0.0
    bank_pct = (inc_bank / income * 100.0) if income else 0.0
    open_orders = [o for o in d.get("orders", []) if o.get("payment_status") != "betaald"]
    te_innen = sum(float(o.get("price_eur") or 0) for o in open_orders)
    return dict(
        inc_cash=round(inc_cash, 2), inc_bank=round(inc_bank, 2),
        inc_wallet=round(inc_wallet, 2),
        income=round(income, 2), cost=round(cost, 2), profit=round(profit, 2),
        topup_total=round(topup, 2), topup_vs_cost=round(topup - cost, 2),
        margin_pct=round(margin, 1), cash_pct=round(cash_pct, 1),
        bank_pct=round(bank_pct, 1), wallet_pct=round(100 - cash_pct - bank_pct, 1),
        gap_pct=round(cash_pct - margin, 1),
        open_orders=len(open_orders), te_innen=round(te_innen, 2))

def monthly_breakdown(d):
    """Per maand: omzet (cash/bank), topups en kosten — de stort-check."""
    rows = {}
    for e in d["entries"]:
        try:
            a = float(e.get("amount_eur") or 0)
        except (TypeError, ValueError):
            continue
        maand = (e.get("ts") or "")[:7]
        if not maand:
            continue
        r = rows.setdefault(maand, dict(cash=0.0, bank=0.0, wallet=0.0, topup=0.0, cost=0.0))
        t = e.get("type")
        if t == "income":
            m = e.get("method")
            if m == "cash":
                r["cash"] += a
            elif m == "basetao":
                r["wallet"] += a
            else:
                r["bank"] += a
        elif t == "cost":
            r["cost"] += a
        elif t == "topup":
            r["topup"] += a
    out = []
    for m in sorted(rows):
        r = rows[m]
        omzet = r["cash"] + r["bank"] + r["wallet"]
        out.append(dict(maand=m, omzet=round(omzet, 2), cash=round(r["cash"], 2),
                        bank=round(r["bank"], 2), topup=round(r["topup"], 2),
                        cost=round(r["cost"], 2), delta=round(r["topup"] - r["cost"], 2)))
    return out

def stats_text_nl():
    s = compute_stats(ledger_load())
    dekking = ("✅ wallet heeft €%g extra" % s["topup_vs_cost"]) if s["topup_vs_cost"] >= 0 \
        else ("⚠️ nog €%g storten" % abs(s["topup_vs_cost"]))
    return (f"📊 Cash €{s['inc_cash']:g} ({s['cash_pct']}%) | "
            f"Bank €{s['inc_bank']:g} ({s['bank_pct']}%) | "
            f"Basetao €{s['inc_wallet']:g} ({s['wallet_pct']}%)\n"
            f"💰 Omzet €{s['income']:g} − kosten €{s['cost']:g} = "
            f"winst €{s['profit']:g} (marge {s['margin_pct']}%)\n"
            f"🏦 Topups €{s['topup_total']:g} vs kosten €{s['cost']:g} → {dekking}")

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
        if t in ("income", "cost", "topup") and amt:
            standaard = {"income": "cash", "cost": "basetao", "topup": "ideal"}[t]
            entry = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": t, "amount_eur": float(amt),
                "method": a.get("method") or standaard,
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
    # Groq whisper-large-v3 eerst (snel + beste NL-kwaliteit); Mistral/Gemini/local als fallback
    if GROQ_API_KEY:
        with open(path, "rb") as f:
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.ogg", f, "audio/ogg")},
                data={"model": "whisper-large-v3", "language": "nl",
                      "temperature": "0", "response_format": "json"},
                timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"groq {r.status_code}: {r.text[:140]}")
        return (r.json().get("text") or "").strip()
    if MISTRAL_API_KEY:
        for lang in ("nl", None):
            with open(path, "rb") as f:
                data = {"model": MISTRAL_ASR_MODEL}
                if lang:
                    data["language"] = lang
                r = requests.post(
                    "https://api.mistral.ai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                    files={"file": ("audio.ogg", f, "audio/ogg")},
                    data=data, timeout=120)
            if r.status_code == 400 and lang:
                continue  # taalparameter niet ondersteund -> opnieuw zonder
            r.raise_for_status()
            return (r.json().get("text") or "").strip()
    if GEMINI_API_KEY:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        model = os.environ.get("GEMINI_ASR_MODEL", "gemini-2.5-flash")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json={"contents": [{"parts": [
                {"text": "Transcribe this audio exactly, in the original language (Dutch). "
                         "Output only the transcription text."},
                {"inline_data": {"mime_type": "audio/ogg", "data": b64}}]}]},
            timeout=120)
        r.raise_for_status()
        d = r.json()
        return ((d.get("candidates") or [{}])[0].get("content", {}).get("parts", [{}])[0]
                .get("text") or "").strip()
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        print("loading whisper model:", WHISPER_MODEL)
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segs, _ = _whisper.transcribe(path, language="nl", beam_size=1)
    return " ".join(s.text.strip() for s in segs).strip()

# ---------------- LLM (OpenAI-compatible) ----------------
SCHEMA_PROMPT = """Je zet Nederlandse berichten om naar bookhoud-acties. Antwoord ONLY met JSON, geen andere tekst:
{"type":"income|cost|topup|order|query|note","amount_eur":number|null,"method":"cash|bank|basetao"|null,"customer":string|null,"items":string|null,"note":string|null}
Regels:
- "cash" = contant geld (physical euro cash). "bank" = Tikkie/overboeking/iDEAL dat je ONTVANGT. "basetao" = directe betaling in de basetao-portemonnee.
- type=topup als Younes zelf geld naar zijn basetao-portemonnee stort (iDEAL-topup voor inkoop).
- amount_eur altijd in euro's (converteer "lek"/"bale"/"lak" naar eur getal).
- type=order als iemand iets bestelt of besteld heeft (klant + items + bedrag) maar er nog geen geld ontvangen is.
- type=query als er om totalen/overzicht gevraagd wordt; type=note als het geen inkomsten/kosten/bestelling/vraag is.
- customer = wie betaalt/gaf opdracht; items = wat is er gekocht/besteld."""

def llm_chat(messages, tools=None, timeout=150):
    """Chat-completions; geeft het volledige assistant-message terug.
    Fallback: eerst met thinking-param, dan minimaal."""
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY ontbreekt")
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    attempts = []
    a1 = {"model": LLM_MODEL, "temperature": 0, "messages": messages}
    if tools:
        a1["tools"] = tools
        a1["tool_choice"] = "auto"
    if LLM_THINKING in ("enabled", "disabled"):
        a1["thinking"] = {"type": LLM_THINKING}
    attempts.append(a1)
    a2 = {"model": LLM_MODEL, "messages": messages}
    if tools:
        a2["tools"] = tools
        a2["tool_choice"] = "auto"
    attempts.append(a2)
    last_err = None
    for payload in attempts:
        try:
            r = requests.post(f"{LLM_BASE_URL}/chat/completions",
                              headers=headers, json=payload, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]
        last_err = f"HTTP {r.status_code}: {r.text[:200]}"
    raise RuntimeError(last_err)

TOOLS = [
    {"type": "function", "function": {
        "name": "add_income",
        "description": "Registreer ontvangen geld van een klant (contant, bank/Tikkie, of direct in de basetao-portemonnee).",
        "parameters": {"type": "object", "properties": {
            "amount_eur": {"type": "number", "description": "bedrag in euro"},
            "method": {"type": "string", "enum": ["cash", "bank", "basetao"]},
            "customer": {"type": "string"},
            "note": {"type": "string"}},
            "required": ["amount_eur", "method"]}}},
    {"type": "function", "function": {
        "name": "add_cost",
        "description": "Registreer een uitgave (bijv. inkoop via basetao of andere kosten).",
        "parameters": {"type": "object", "properties": {
            "amount_eur": {"type": "number"}, "note": {"type": "string"}},
            "required": ["amount_eur"]}}},
    {"type": "function", "function": {
        "name": "add_topup",
        "description": "Registreer dat Younes geld heeft gestort naar zijn basetao-portemonnee "
                       "(iDEAL-topup). Geen inkomsten — een storting om inkoop te betalen.",
        "parameters": {"type": "object", "properties": {
            "amount_eur": {"type": "number"}, "note": {"type": "string"}},
            "required": ["amount_eur"]}}},
    {"type": "function", "function": {
        "name": "create_order",
        "description": "Nieuwe order aanmaken voor een klant (nog niet betaald).",
        "parameters": {"type": "object", "properties": {
            "customer": {"type": "string"}, "items": {"type": "string"},
            "price_eur": {"type": "number"}},
            "required": ["customer"]}}},
    {"type": "function", "function": {
        "name": "update_order",
        "description": "Order bijwerken (betaalstatus, orderstatus, klant, items, prijs). "
                       "Orderstatus: interesse, info_gevraagd, prijs_gegeven, wacht_op_antwoord, "
                       "te_bestellen, besteld, onderweg, binnen, verpakken, klaar, geen_interesse. "
                       "Betaalstatus: nog_niet_betaald, deels_betaald, betaald.",
        "parameters": {"type": "object", "properties": {
            "num": {"type": "integer"},
            "order_status": {"type": "string"},
            "payment_status": {"type": "string"},
            "payment_method": {"type": "string", "enum": ["cash", "bank", "basetao"]},
            "customer": {"type": "string"}, "items": {"type": "string"},
            "price_eur": {"type": "number"}},
            "required": ["num"]}}},
    {"type": "function", "function": {
        "name": "delete_order",
        "description": "Order permanent verwijderen.",
        "parameters": {"type": "object", "properties": {"num": {"type": "integer"}},
                       "required": ["num"]}}},
    {"type": "function", "function": {
        "name": "list_orders",
        "description": "Laatste orders met order- en betaalstatus.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_stats",
        "description": "Totalen: cash vs basetao inkomsten, kosten, winstmarge, open orders, te innen.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "basetao_status",
        "description": "Live basetao-portemonneesaldo en ordertellers.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "zoek_qc",
        "description": "Zoek producten met QC-foto's op doppel.fit (rep-sourcing). Geeft top-items "
                       "met titel, prijs, verkoper en QC-fotolinks. Gebruik dit als een klant vraagt "
                       "wat er leverbaar is of als Younes iets moet sourcen, bijv. 'groene dunks maat 42'. "
                       "Vereist dat de doppel-bridge (userscript in Chrome van Younes) aan staat; "
                       "anders krijg je een melding dat de bridge offline is.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "zoekterm, bijv. 'nike dunk low panda'"},
            "count": {"type": "integer", "description": "aantal items (1-5), standaard 3"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "chat_status",
        "description": "Werk het klantoverzicht bij voor DEZE chat — de 'stand van zaken' die Younes "
                       "leest om niks te vergeten. Roep dit aan aan het eind van elke klant-ronde "
                       "met wat je op dit moment weet (of wat is veranderd).",
        "parameters": {"type": "object", "properties": {
            "klantnaam": {"type": "string", "description": "voornaam zoals de klant die gaf"},
            "gezocht": {"type": "string", "description": "wat de klant wil: merk/model/kleur/bedrukking"},
            "maat": {"type": "string", "description": "bekende maat, of 'lengte+gewicht: 178cm/70kg', of onbekend"},
            "prijs_afgesproken": {"type": "number", "description": "afgesproken prijs in euro; weglaten als er nog geen prijs is"},
            "prijs_gegeven": {"type": "boolean", "description": "true als de klant al een concrete prijs heeft gekregen (van Younes of een vaste prijs)"},
            "foto_ontvangen": {"type": "boolean", "description": "true als er een productfoto of link is"},
            "ontbreekt": {"type": "string", "description": "wat je nog nodig hebt, bijv. 'foto, maat (lengte+gewicht)' of 'leeg'"},
            "status": {"type": "string", "description": "kort: nieuw / wacht_op_info / prijs_gegeven / klaar_voor_bestelling / besteld / afgerond"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "vraag_younes",
        "description": "Stel een vraag rechtstreeks aan Younes in zijn eigen WhatsApp-chat "
                       "(bijv. een prijs die je niet weet, of een vraag waar je het antwoord "
                       "niet van weet). Je antwoord gaat als melding naar hem; zijn antwoord "
                       "wordt automatisch naar de klant doorgestuurd. Je krijgt eventueel een "
                       "BEKENDE PRIJS terug die je meteen mag noemen.",
        "parameters": {"type": "object", "properties": {
            "vraag": {"type": "string", "description": "korte vraag, bijv. 'prijs van: Nike Dunk Low groen maat 43'"},
            "klant": {"type": "string", "description": "naam van de klant die vroeg"}},
            "required": ["vraag"]}}},
    {"type": "function", "function": {
        "name": "vaste_prijzen",
        "description": "Haal de actuele prijslijst op (door Younes vastgesteld). "
                       "Gebruik ALLÉÉN deze prijzen naar klanten.",
        "parameters": {"type": "object", "properties": {},
                       "required": []}}},
]

SYSTEM_AGENT = """Je bent Rep Agent, de boekhoudmaat van Younes: hij verkoopt reps (kleding, sneakers, sets) via Snapchat, WhatsApp en Telegram en inkoopt via basetao. Je praat in zijn taal: kort, casual, Nederlands, max ~4 regels, emoji's zijn oké.

Werkwijze:
- Voeg geld, kosten en orders direct toe of werk ze bij met de tools. Vraag niet om toestemming voor iets wat duidelijk is.
- Als een klant betaalt voor een bestaande order: markeer die order betaald (update_order met num als je hem weet) EN registreer het geld (add_income).
- Als bedrag, methode (cash vs basetao) of product onduidelijk is: stel maximaal één korte doorvraag. Probeer verder zelf in te schatten.
- Maten: vraag lengte/gewicht als iemand onduidelijk is over maat.
- Bij sourcing-vragen ("heb je X", "wat kost Y", klant zoekt iets): gebruik zoek_qc en geef de beste matches kort met prijs en QC-link.
- Bij "hoeveel/wat is mijn stand"-vragen: gebruik get_stats (en basetao_status) en vat samen.
- Stort Younes zelf geld naar zijn basetao-wallet (iDEAL)? Gebruik add_topup. Tikkie/overboeking van een klant = add_income met method "bank".
- Vermeld aan het eind kort wat je hebt gedaan of wat openstaat."""

SYSTEM_WA_KLANT = """Je bent de WhatsApp-assistent van YZ Shop van Younes en je APPT ALS YOUNES ZELF.

STIJL — dit is hoe Younes écht appt (uit zijn eigen oude berichten):
- Kort, los, straight maar vriendelijk. Geen nette zinnen, kleine typfouten mogen.
- Zijn eigen woorden: "yo", "bro", "fam", "setje", "broekje", "eu" i.p.v. euro, "ik heb em liggen of ik kan eraan komen", "ik laat je weten", "ik kom erop terug", "stuur maar even", "dan heb je em".
- Zijn echte voorbeeldzinnen: "yo kan je aan dit komen ja of nee" · "die is 30eu" · "2-3 weken dan heb je em" · "die is nice" · "ik pak em voor je".
- GEEN verkooppraat, geen "beste klant", geen lange zinnen, geen perfecte punctatie.

HARD PRIVATE: je geeft NOOIT informatie over andere klanten, andere bestellingen, omzet, voorraad of boekhouding. Vraagt een klant "wat heb ik besteld?" → noem ALLÉÉN wat jij in dit gesprek zelf genoteerd hebt (staat in chat_status); ken je niks, zeg dan gewoon "wat zocht je ook alweer?". Boekhoudtools zijn voor jou geblokkeerd en je noemt dat nooit.

PRIJZEN: roep vaste_prijzen aan voor de actuele prijslijst en noem ALLÉÉN prijzen uit die lijst (of een prijs die Younes in dit gesprek al zelf noemde — chat_status prijs_afgesproken geldt altijd). Staat iets niet in de lijst? Dan nooit een bedrag verzinnen: "die moet ik even voor je checken, ik kom erop terug 👍" en vraag_younes gebruiken.
Werkwijze:
- KORT: max 2-3 korte zinnen per bericht. Geen lijsten, geen prijslijsten, geen lange uitleg — ook niet als de klant doorvraagt ("wat ga je kijken?" → "even checken wat er kan 👍").
- EINDE VAN DE CHAT: je hoeft NIET het laatste woord. Heeft de klant genoeg gezegd, of zeggen ze gewoon iets waar je niks op hoeft te vragen? Reageer dan kort en KLAAR: "safi 👍 ik pak em op", "gezet bro", "ik laat je weten". Niet elk bericht afsluiten met een vraag — dat is opdringerig. Vraag ALLÉÉN het punt dat in chat_status.ontbreekt nog écht mist; de rest is al gezegd.
- DENK ALS YOUNES bij elk bericht: is dit een normale chat? Weet ik wat er aan de hand is? Heb ik genoeg info? Zo ja → kort bevestigen en klaar. Zo nee → alleen het ontbrekende punt vragen, niets opnieuw.
- NIET IN HERHALING: het gesprek hieronder EN chat_status bevatten alles wat de klant al verteld heeft (naam, wat hij zoekt, maat, foto, prijs). Lees dat EERST terug — vraag NOOIT opnieuw wat er al in staat, ook niet als het even terugzoeken is ("Ik heb het hier staan 👍 je zocht X in maat Y toch?"). Zegt de klant "dat heb ik al gestuurd"? Dan staat het in het gesprek: bevestig wat er staat i.p.v. opnieuw vragen. Een vraag eenmaal gesteld = wachten op het antwoord.
- FOTO'S: staat er [foto] in het gesprek, of foto_ontvangen=true in chat_status? Dan is er AL een foto gestuurd — vraag dan NOOIT nog eens om een foto.
- Taal: je antwoordt ALTIJD in het Nederlands, ook als de klant Engels of een andere taal schrijft. Alleen Engels als de klant er expliciet om vraagt.
- INTAKE — voordat iets besteld kan worden heb je ALTIJD deze punten nodig. Vraag ze stap voor stap (1-2 punten per bericht) en herhaal kort wat de klant al gaf:
  1. Wát precies: merk/model/kleur — en een PRODUCTFOTO of link ("stuur even een pic van wat je wilt fam, dan pak ik precies die").
  2. MAAT: kleding = lengte + gewicht ("hoe lang ben je en hoeveel weeg je? Dan heb ik je maat"); schoenen = schoenmaat.
  3. Voetbalshirt/set: bedrukking = naam + rugnummer.
  4. Eenmalig, vroeg in het gesprek: "zet even je verdwijnende berichten uit in deze chat, dan blijft ons gesprek staan."
  Klanten moeten SPECIFIEK zijn: vaag ("wil graag zoiets") = doorvragen tot je het exact kunt opschrijven.
- PRIJZEN — cruciaal: geef NOOIT een bedrag zonder vaste_prijzen te hebben geroepen in dit gesprek. Alles wat niet in die lijst staat (ALO, schoenen, tassen, hoodies, brillen, jassen, andere sneakers...) = NOOIT een bedrag bedenken. Roep vraag_younes aan ("prijs van: <product> maat <maat>") — krijg je een BEKENDE PRIJS terug dan mag je die meteen noemen; anders zeg je: "ik check het even bij Younes, ik kom erop terug 👍". Younes' antwoord komt automatisch bij de klant, daar hoef je niet meer naar om te kijken. Heeft Younes al een prijs genoemd in dit gesprek (chat_status: prijs_afgesproken)? Dan is DIE leidend en herhaal je die exact.
- WEET JE HET ANTWOORD NIET (random vragen, andere kleur kunnen, levertijd van iets specifieks, etc.)? Roep vraag_younes aan met de korte vraag — Younes antwoordt en het gaat automatisch naar de klant. Zeg zelf alleen: "dat check ik even voor je 👍".
- Wil een klant iets specifieks (merk/model/kleur)? Gebruik zoek_qc en toon de beste match kort (max 2 regels + foto). De prijzen uit zoek_qc zijn INKOOPprijzen — NOOIT tegen de klant noemen. Geen resultaten? "Laat ik even kijken, ik hoor zo van je."
- Is de intake compleet (wat + foto + maat + bedrukking + prijs duidelijk)? Bevestig kort dat je het bij Younes inwerkt en maak een create_order aan (prijs alleen invullen als die afgesproken is).
- Roep aan het eind van ELKE klant-ronde chat_status aan met wat je nu weet (klantnaam, gezocht, maat, prijs, foto, ontbreekt, status).
- Beloof nooit leverdatums buiten 2-3 weken. Blijf beleefd ook als de klant bot is.
- Noem nooit interne tools, foutmeldingen of technische details tegen klanten. Als iets niet lukt: "ik laat zo wat horen"."""

_chatmem = {}

def agent_reply(chat_id, text):
    """Interactieve agent-loop met toolgebruik en gespreksgeheugen per chat.
    WhatsApp-chats (wa:) praten met klanten -> ander brein dan boekhouding."""
    hist = _chatmem.setdefault(chat_id, [])
    hist.append({"role": "user", "content": text})
    system = SYSTEM_WA_KLANT if str(chat_id).startswith("wa:") else SYSTEM_AGENT
    messages = [{"role": "system", "content": system}] + hist[-16:]
    answer = None
    for _ in range(5):
        msg = llm_chat(messages, tools=TOOLS)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = (msg.get("content") or "").strip()
            break
        for c in calls:
            fn = c.get("function") or {}
            result = run_tool(fn.get("name"), fn.get("arguments") or "{}", chat_key=chat_id)
            messages.append({"role": "tool", "tool_call_id": c.get("id"),
                             "content": str(result)[:800]})
    if not answer:
        answer = stats_text_nl()
    hist.append({"role": "assistant", "content": answer})
    del hist[:-40]
    return answer

def run_tool(name, args_json, chat_key=None):
    try:
        a = json.loads(args_json) if isinstance(args_json, str) else (args_json or {})
    except Exception:  # noqa: BLE001
        a = {}
    # PRIVACY: in klantgesprekken (WhatsApp) zijn alle boekhoudtools en klantgegevens
    # van anderen hard geblokkeerd. Alleen bestellen, zoeken en de eigen chat-status.
    if chat_key and chat_key.startswith("wa:") and name not in (
            "create_order", "zoek_qc", "chat_status"):
        return "GEBLOKKEERD: klantgesprekken mogen nooit boekhouding of gegevens van andere klanten zien."
    try:
        if name == "zoek_qc":
            return tool_zoek_qc(a.get("query"), a.get("count"), chat_key=chat_key)
        if name == "add_income":
            with _ledlock:
                d = ledger_load()
                d.setdefault("entries", []).append({
                    "ts": _now(), "type": "income", "amount_eur": float(a.get("amount_eur") or 0),
                    "method": a.get("method") or "cash", "customer": a.get("customer"),
                    "note": a.get("note") or "telegram", "source": "telegram"})
                ledger_save(d)
                s = compute_stats(d)
                hf_sync_up()
            return (f"inkomsten €{a.get('amount_eur')} ({a.get('method')}) geboekt. "
                    f"Stand: cash {s['cash_pct']}% | marge {s['margin_pct']}%")
        if name == "add_cost":
            with _ledlock:
                d = ledger_load()
                d.setdefault("entries", []).append({
                    "ts": _now(), "type": "cost", "amount_eur": float(a.get("amount_eur") or 0),
                    "method": "basetao", "note": a.get("note") or "telegram",
                    "source": "telegram"})
                ledger_save(d)
                s = compute_stats(d)
                hf_sync_up()
            return f"kosten €{a.get('amount_eur')} geboekt, marge nu {s['margin_pct']}%"
        if name == "add_topup":
            with _ledlock:
                d = ledger_load()
                d.setdefault("entries", []).append({
                    "ts": _now(), "type": "topup", "amount_eur": float(a.get("amount_eur") or 0),
                    "method": "ideal", "note": a.get("note") or "iDEAL topup basetao",
                    "source": "telegram"})
                ledger_save(d)
                s = compute_stats(d)
                hf_sync_up()
            return (f"topup €{a.get('amount_eur')} geboekt | gestort €{s['topup_total']} vs "
                    f"kosten €{s['cost']} (verschil €{s['topup_vs_cost']})")
        if name == "chat_status":
            if not chat_key or not chat_key.startswith("wa:"):
                return "chat_status werkt alleen in WhatsApp-chats"
            with _ledlock:
                d = ledger_load()
                c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(chat_key[3:], {})
                for f in ("klantnaam", "gezocht", "maat", "ontbreekt", "status"):
                    if a.get(f) is not None:
                        c[f] = str(a[f])[:300]
                if a.get("prijs_afgesproken") is not None:
                    try:
                        c["prijs_afgesproken"] = float(a["prijs_afgesproken"])
                    except (TypeError, ValueError):
                        pass
                for f in ("prijs_gegeven", "foto_ontvangen"):
                    if a.get(f) is not None:
                        c[f] = bool(a[f])
                c["bijgewerkt"] = _now()
                ledger_save(d)
                hf_sync_up()
            return "klantoverzicht bijgewerkt"
        if name == "vraag_younes":
            if not chat_key or not chat_key.startswith("wa:"):
                return "vraag_younes werkt alleen in WhatsApp-chats"
            vraag = str(a.get("vraag") or "?").strip()[:300]
            with _ledlock:
                d = ledger_load()
                pend = d.setdefault("whatsapp", {}).setdefault("pending", [])
                for p in pend:
                    if not p.get("answered") and p.get("chat_key") == chat_key and p.get("vraag") == vraag:
                        return "deze vraag staat al aan Younes voorgelegd — zeg de klant dat je hem terugappt"
                n = 1 + max([p.get("n", 0) for p in pend] or [0])
                c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(chat_key[3:], {})
                naam = str(a.get("klant") or c.get("klantnaam") or c.get("naam")
                           or ("+" + chat_key[3:]))[:60]
                pend.append({"n": n, "chat_key": chat_key, "klant": naam, "vraag": vraag,
                             "ts": _now(), "answered": False})
                ledger_save(d)
                hf_sync_up()
            bek = ""
            for prod, pr in (d.get("whatsapp", {}).get("prijzen") or {}).items():
                if prod and prod in vraag.lower():
                    bek += f" BEKENDE PRIJS ({prod}): €{pr:g} — deze mag je WEL noemen."
            _stuur_via_bridge(WA_SELF_JID,
                              f"❓ {naam} vroeg: {vraag}\n→ antwoord met: antwoord {n} <je antwoord>")
            return ("Vraag staat bij Younes." + bek +
                    " Zeg tegen de klant: 'ik check het even, ik kom erop terug 👍'")
        if name == "vaste_prijzen":
            with _ledlock:
                d = ledger_load()
                p = d.setdefault("whatsapp", {}).setdefault("vaste_prijzen", {})
                if not p:
                    p["voetbalshirt custom (naam + rugnummer)"] = 30.0
                    p["set (shirt + broekje)"] = 40.0
                    ledger_save(d)
            regels = [f"- {k}: €{v:g}" for k, v in p.items()]
            regels.append("Levertijd: 2-3 weken. Alleen deze prijzen noemen; iets anders = checken bij Younes.")
            return "\n".join(regels)
        if name == "create_order":
            with _ledlock:
                d = ledger_load()
                o = order_add(d, a.get("customer"), a.get("items"), a.get("price_eur") or 0,
                              note="telegram")
                ledger_save(d)
                hf_sync_up()
            return (f"order #{o['num']} aangemaakt: {o['customer']} — {o['items'] or '?'} "
                    f"€{o['price_eur']:g} (te bestellen, nog niet betaald)")
        if name == "update_order":
            with _ledlock:
                d = ledger_load()
                o = order_update(d, int(a.get("num")), order_status=a.get("order_status"),
                                 payment_status=a.get("payment_status"),
                                 payment_method=a.get("payment_method"),
                                 customer=a.get("customer"), items=a.get("items"),
                                 price_eur=a.get("price_eur"))
                ledger_save(d)
                hf_sync_up()
            if not o:
                return "order niet gevonden"
            return (f"order #{o['num']} bijgewerkt: 📦{o['order_status']} 💶{o['payment_status']}"
                    + (f" ({o['payment_method']})" if o.get("payment_method") not in (None, "onbekend") else ""))
        if name == "delete_order":
            with _ledlock:
                d = ledger_load()
                before = len(d.get("orders", []))
                d["orders"] = [o for o in d.get("orders", []) if o.get("num") != int(a.get("num"))]
                gone = len(d["orders"]) < before
                ledger_save(d)
                hf_sync_up()
            return "order verwijderd" if gone else "order niet gevonden"
        if name == "list_orders":
            with _ledlock:
                d = ledger_load()
            orders = d.get("orders", [])[-10:]
            if not orders:
                return "geen orders"
            return "\n".join(f"#{o['num']} {o['customer']} — {o['items'] or '?'} €{o['price_eur']:g} — "
                             f"📦{o['order_status']} 💶{o['payment_status']}" for o in orders)
        if name == "get_stats":
            with _ledlock:
                d = ledger_load()
            s = compute_stats(d)
            return (f"cash €{s['inc_cash']} ({s['cash_pct']}%) | bank €{s['inc_bank']} "
                    f"({s['bank_pct']}%) | inkomsten €{s['income']} | kosten €{s['cost']} | "
                    f"winst €{s['profit']} (marge {s['margin_pct']}%) | topups €{s['topup_total']} "
                    f"vs kosten €{s['cost']} (€{s['topup_vs_cost']}) | open orders {s['open_orders']} "
                    f"(te innen €{s['te_innen']})")
        if name == "basetao_status":
            w = basetao_wallet(max_age=60) or {}
            c = w.get("counters") or {}
            if not w.get("logged_in"):
                return "basetao niet bereikbaar of cookie verlopen"
            return (f"saldo ¥{w.get('balance_cny') or '?'} | ordered {c.get('Pending', '?')} | "
                    f"arrived {c.get('Arrived', '?')} | shipped {c.get('Shipped', '?')} | "
                    f"searching {c.get('Searching', '?')} | pakketten ontvangen {c.get('Received', '?')}")
        return f"onbekende tool {name}"
    except Exception as e:  # noqa: BLE001
        print("tool error:", name, e)
        return f"tool-fout: {e}"

def llm_parse(text):
    if not LLM_API_KEY:
        return {"type": "note", "note": text}
    content = llm_chat([{"role": "system", "content": SCHEMA_PROMPT},
                        {"role": "user", "content": text}])
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0) if m else content)

EXTRACT_PROMPT = """Haal uit deze Basetao-orderlijst alle producten. Antwoord ONLY met een JSON-array, geen andere tekst:
[{"order_id":"1936771","title":"productnaam","price_cny":123.45}]
- order_id = het Basetao-ordernummer bij het product (alleen cijfers)
- title = de product/mdl-omschrijving
- price_cny = de prijs in CNY-nummer (null als niet vermeld)
Sla dubbelloopse kopregels/paginering over. Geen producten? antwoord []"""

def llm_extract_products(text):
    msg = llm_chat([{"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": text[:14000]}], timeout=180)
    content = msg.get("content") or ""
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
    """Basetao-sessie testen + saldo/tellers uit de server-HTML.
    CF-prerendering levert de data willekeurig mee -> tot 4x proberen."""
    headers = {
        "Cookie": BASETAO_COOKIE,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
        "Referer": "https://www.basetao.com/",
        "Accept": "text/html",
    }
    logged_in, balance, counters, http_code = False, None, {}, 0
    for attempt in range(4):
        try:
            r = requests.get(
                "https://www.basetao.com/best-taobao-agent-service/my_account/welcome.html",
                headers=headers, timeout=30,
                params={"r": str(int(time.time() * 1000)) + str(attempt)})
        except Exception as e:  # noqa: BLE001
            print("basetao fetch error:", e)
            break
        http_code = r.status_code
        t = r.text
        logged_in = "Welcome back" in t
        m = re.search(r'bi-currency-yen[\s\S]{0,150}?>\s*([\d.,]+)\s*<', t)
        balance = m.group(1) if m else None
        counters = dict(re.findall(
            r'id="(Ordered|Arrived|Cancelled|Shipped|Searching|Received|Pending)"'
            r'[\s\S]{0,300}?badge[^>]*>\s*(\d+)\s*</span>', t))
        if logged_in and (balance or counters):
            break
        time.sleep(1.2)
    return {"logged_in": logged_in, "http": http_code,
            "balance_cny": balance, "counters": counters,
            "note": "saldo/tellers komen mee wanneer basetao ze in de HTML serveert"}

def handle_update(msg):
    chat_id = msg["chat"]["id"]
    media = msg.get("voice") or msg.get("audio") or msg.get("document")
    if media:
        tg("sendChatAction", chat_id=chat_id, action="typing")
        path = tg_download(media["file_id"])
        if not path:
            tg("sendMessage", chat_id=chat_id, text="❌ Kon voicebestand niet ophalen.")
            return
        audio_bytes = b""
        try:
            text = transcribe(path)
            with open(path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:  # noqa: BLE001
            print("asr error:", e)
            detail = str(e)[:180] or "onbekend"
            tg("sendMessage", chat_id=chat_id,
               text=f"❌ Spraakherkenning mislukt.\n🔧 {detail}\nProbeer opnieuw of typ het.")
            return
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if not text:
            tg("sendMessage", chat_id=chat_id, text="❓ Geen spraak herkend.")
            return
        _voice_ctx[str(chat_id)] = media.get("file_id")
        voice_note_opslaan(text, "telegram", chat_key=str(chat_id),
                           file_id=media.get("file_id"), audio_bytes=audio_bytes)
        tg("sendMessage", chat_id=chat_id, text=f"🎙️ \"{text}\"")
    elif msg.get("text"):
        text = msg["text"].strip()
    else:
        return
    low = text.lower()
    if low.startswith("/basetao") or low.startswith("/orders") or low.startswith("/order ") \
            or low.startswith("/status ") or low.startswith("/betaald") or low.startswith("/klant "):
        return handle_command(chat_id, text)
    tg("sendChatAction", chat_id=chat_id, action="typing")
    try:
        answer = agent_reply(chat_id, text)
    except Exception as e:  # noqa: BLE001
        print("agent error:", e)
        tg("sendMessage", chat_id=chat_id, text="❌ Dat ging mis, probeer het nog eens.")
        return
    tg("sendMessage", chat_id=chat_id, text=answer)

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
        m = re.match(r"/betaald\s+#?(\d+)(?:\s+(cash|bank|basetao))?", low)
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
    # startpositie herstellen uit het kasboek zodat herstarts geen herhaling geven
    try:
        with _ledlock:
            offset = int(ledger_load().get("tg_offset") or 0)
    except Exception:  # noqa: BLE001
        offset = 0
    print("telegram polling started, token set:", bool(TELEGRAM_TOKEN), "| offset:", offset)
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
                    with _ledlock:
                        d = ledger_load()
                        d["tg_offset"] = offset
                        ledger_save(d)
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
<title>YZ Shop — Admin</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0f1716;color:#e8efec;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px}.sub{color:#8aa39c;font-size:13px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;max-width:980px}
.card{background:#182522;border:1px solid #24382f;border-radius:12px;padding:16px}
.k{color:#8aa39c;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.v{font-size:26px;font-weight:700;margin-top:6px}
.bar{height:14px;border-radius:7px;overflow:hidden;display:flex;margin-top:10px;background:#0c1210}
.bar span{height:100%}.cash{background:#4caf7d}.bank{background:#3d7dd8}.wallet{background:#8a6fd1}
.gap-ok{color:#4caf7d}.gap-bad{color:#e0a13d}
.v.ok{color:#4caf7d}.v.bad{color:#e0a13d}.v.lime{color:#d7ff3f}
h2{font-size:16px;margin:30px 0 0}
.legend{display:flex;gap:14px;margin-top:10px;font-size:12px;color:#c8d6d1;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
tr.tot td{font-weight:700;border-top:2px solid #2b463c}
.ok{color:#4caf7d}.bad{color:#e0a13d}
table{width:100%;max-width:980px;border-collapse:collapse;margin-top:22px;font-size:13px}
td,th{padding:7px 9px;border-bottom:1px solid #1e2f28;text-align:left}
th{color:#8aa39c;font-weight:600}form{margin-top:26px;max-width:980px;background:#182522;
border:1px solid #24382f;border-radius:12px;padding:16px;display:grid;
grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
input,select{background:#0c1210;border:1px solid #2b463c;color:#e8efec;border-radius:8px;padding:8px;width:100%}
button{background:#d7ff3f;border:0;color:#10130f;border-radius:8px;padding:10px;font-weight:700;cursor:pointer}
.note{color:#8aa39c;font-size:12px;margin-top:18px;max-width:980px}
</style></head><body>
<h1>YZ SHOP — cash &amp; bank</h1>
<div class="sub">live · verversen elke 60s · regel: alleen inkoop+verzending storten → alle winst blijft cash</div>
<div class="grid">
<div class="card"><div class="k">Cash ontvangen</div><div class="v ok">€__CASH__</div>
<div class="k" style="margin-top:6px">__CASH_PCT__% van omzet</div></div>
<div class="card"><div class="k">Bank ontvangen (tikkie/ideal)</div><div class="v">€__BANK__</div>
<div class="k" style="margin-top:6px">__BANK_PCT__% van omzet</div></div>
<div class="card"><div class="k">Basetao-inkomsten</div><div class="v">€__WALLET__</div>
<div class="k" style="margin-top:6px">__WALLET_PCT__% van omzet</div></div>
<div class="card"><div class="k">Omzet</div><div class="v lime">€__INCOME__</div>
<div class="k" style="margin-top:6px">winst €__PROFIT__ · marge __MARGIN__%</div></div>
<div class="card"><div class="k">Kosten (inkoop + verzending)</div><div class="v">€__COST__</div>
<div class="k" style="margin-top:6px">gebetaald uit basetao-wallet</div></div>
<div class="card"><div class="k">Open orders</div><div class="v">__OPENORDERS__</div>
<div class="k" style="margin-top:6px">te innen: €__TEINNEN__</div></div>
</div>
<div class="grid" style="margin-top:14px">
<div class="card wide"><div class="k">Cash → bank ratio</div>
<div class="bar" style="height:20px"><span class="cash" style="width:__CASH_PCT__%"></span><span class="bank" style="width:__BANK_PCT__%"></span><span class="wallet" style="width:__WALLET_PCT__%"></span></div>
<div class="legend"><span><span class="dot" style="background:#4caf7d"></span>cash __CASH_PCT__%</span><span><span class="dot" style="background:#3d7dd8"></span>bank __BANK_PCT__%</span><span><span class="dot" style="background:#8a6fd1"></span>basetao __WALLET_PCT__%</span><span>doel: cash% ≥ marge __MARGIN__% · verschil __GAP__ pp __GAPCLS_TXT__</span></div></div>
<div class="card wide"><div class="k">Stort-check — topups vs kosten</div>
<div class="v __TOPUP_CLS__">__TOPUP_TXT__</div>
<div class="k" style="margin-top:6px">€__TOPUP__ gestort via iDEAL vs €__COST__ product+verzending · __STORT_ADVIES__</div></div>
<div class="card"><div class="k">Basetao saldo (live)</div><div class="v">¥__BTBAL__</div>
<div class="k" style="margin-top:6px">__BTCNT__</div></div>
</div>
<h2>📅 Per maand — stort-check</h2>
<table><tr><th>Maand</th><th>Omzet</th><th>Cash</th><th>Bank</th><th>Topups</th><th>Kosten</th><th>Topup − kosten</th></tr>
__MROWS__
</table>
<h2>📋 Orders &amp; betalingen</h2>
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
<select id="f_type"><option value="income">inkomsten</option><option value="cost">kosten</option><option value="topup">topup (storting)</option></select>
<select id="f_method"><option value="cash">cash</option><option value="bank">bank/tikkie</option><option value="basetao">basetao</option></select>
<input id="f_customer" placeholder="klant">
<input id="f_note" placeholder="notitie">
<button>Toevoegen</button></form>
<form onsubmit="addNote(event)">
<input id="f_note2" placeholder="notitie / memo (bv. transcript of afspraak)">
<button>Notitie opslaan</button></form>
<script>const K=new URLSearchParams(location.search).get('key')||(document.cookie.split('; ').find(r=>r.startsWith('key='))||'').slice(4)||'';
if(K)document.cookie='key='+K+';path=/;max-age=31536000';
async function add(e){e.preventDefault();const g=i=>document.getElementById(i).value;
const t=g('f_type');const b={type:t,amount_eur:parseFloat(g('f_amount')),
customer:g('f_customer')||null,note:g('f_note')||null};
b.method=(t==='topup')?'ideal':g('f_method');
const r=await fetch('/api/entry',{method:'POST',headers:{'Content-Type':'application/json','X-Access-Code':K},
body:JSON.stringify(b)});
r.ok?location.reload():alert('mislukt');}
async function addNote(e){e.preventDefault();const n=document.getElementById('f_note2').value.trim();
if(!n)return;const r=await fetch('/api/note',{method:'POST',headers:{'Content-Type':'application/json','X-Access-Code':K},body:JSON.stringify({note:n})});
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
<div class="note">topup = iDEAL-storting naar je basetao-wallet (geen omzet) · bank = tikkie/overboeking ontvangen · checkout-links worden automatisch als topup geboekt · stemnotities worden bewaard in de HF-dataset (map audio/).</div>
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
        maanden = monthly_breakdown(d)
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
    os_opts = "".join(f'<option value="{x}">{x}</option>' for x in ORDER_STATUSES)
    ps_opts = "".join(f'<option value="{x}">{x}</option>' for x in PAYMENT_STATUSES)
    gap_txt = "✅" if abs(s["gap_pct"]) < 5 else "⚠️"
    if s["topup_vs_cost"] >= 0:
        topup_txt = f"✅ gedekt (+€{s['topup_vs_cost']:g})"
        topup_cls = "ok"
        advies = "wallet heeft voorraad — niks bijstorten nodig"
    else:
        topup_txt = f"⚠️ te kort (−€{abs(s['topup_vs_cost']):g})"
        topup_cls = "bad"
        advies = f"stort nog €{abs(s['topup_vs_cost']):g}, dan blijft al je cash winst"
    mrows = []
    if maanden:
        t_om = t_c = t_b = t_t = t_k = 0.0
        for r in maanden:
            dcls = "ok" if r["delta"] >= 0 else "bad"
            ds = f"+€{r['delta']:g}" if r["delta"] >= 0 else f"−€{abs(r['delta']):g}"
            mrows.append(f"<tr><td>{r['maand']}</td><td>€{r['omzet']:g}</td><td>€{r['cash']:g}</td>"
                         f"<td>€{r['bank']:g}</td><td>€{r['topup']:g}</td><td>€{r['cost']:g}</td>"
                         f"<td class='{dcls}'>{ds}</td></tr>")
            t_om += r["omzet"]
            t_c += r["cash"]
            t_b += r["bank"]
            t_t += r["topup"]
            t_k += r["cost"]
        t_delta = round(t_t - t_k, 2)
        tds = f"+€{t_delta:g}" if t_delta >= 0 else f"−€{abs(t_delta):g}"
        mrows.append(f"<tr class='tot'><td>totaal</td><td>€{round(t_om, 2):g}</td>"
                     f"<td>€{round(t_c, 2):g}</td><td>€{round(t_b, 2):g}</td>"
                     f"<td>€{round(t_t, 2):g}</td><td>€{round(t_k, 2):g}</td><td>{tds}</td></tr>")
    return (DASH
            .replace("__CASH__", f"{s['inc_cash']:g}")
            .replace("__BANK__", f"{s['inc_bank']:g}")
            .replace("__WALLET__", f"{s['inc_wallet']:g}")
            .replace("__CASH_PCT__", str(s["cash_pct"]))
            .replace("__BANK_PCT__", str(s["bank_pct"]))
            .replace("__WALLET_PCT__", str(s["wallet_pct"]))
            .replace("__INCOME__", f"{s['income']:g}")
            .replace("__PROFIT__", f"{s['profit']:g}")
            .replace("__MARGIN__", str(s["margin_pct"]))
            .replace("__COST__", f"{s['cost']:g}")
            .replace("__GAP__", str(s["gap_pct"]))
            .replace("__GAPCLS_TXT__", gap_txt)
            .replace("__TOPUP__", f"{s['topup_total']:g}")
            .replace("__TOPUP_TXT__", topup_txt)
            .replace("__TOPUP_CLS__", topup_cls)
            .replace("__STORT_ADVIES__", advies)
            .replace("__OPENORDERS__", str(s["open_orders"]))
            .replace("__TEINNEN__", f"{s['te_innen']:g}")
            .replace("__BTBAL__", str(btbal))
            .replace("__BTCNT__", btc)
            .replace("__MROWS__", "\n".join(mrows) or '<tr><td colspan="7">— nog geen bedragen —</td></tr>')
            .replace("__OROWS__", "\n".join(orows) or '<tr><td colspan="9">— nog geen orders —</td></tr>')
            .replace("__OSOPT__", os_opts)
            .replace("__PSOPT__", ps_opts)
            .replace("__ROWS__", "\n".join(rows) or "<tr><td colspan=6>—</td></tr>"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def access_gate(request: Request, call_next):
    """Site is openbaar; alleen beheer-routes vereisen de toegangscode."""
    if request.method == "OPTIONS":  # CORS-voorvraag: altijd doorlaten
        return await call_next(request)
    path = request.url.path
    gated = (path.startswith("/admin") or path.startswith("/api")
             or path.startswith("/stats") or path.startswith("/orders")
             or path.startswith("/basetao") or path.startswith("/docs")
             or path.startswith("/openapi") or path.startswith("/whatsapp"))
    if ACCESS_CODE and gated:
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
    wa_backup_down()
    if WA_TAKEOVER_HOURS <= 0:
        # auto-pauze staat uit: ruim oude pauzes op zodat chats direct weer werken
        try:
            with _ledlock:
                d = ledger_load()
                paused = d.get("whatsapp", {}).get("paused", {})
                if paused:
                    paused.clear()
                    ledger_save(d)
                    print("oude whatsapp-pauzes gewist (takeover uit)")
        except Exception as e:  # noqa: BLE001
            print("pauzes wissen mislukt:", e)
      threading.Thread(target=poll_loop, daemon=True).start()

      def _herinneringen():
          # Elke 4 uur: wat moet Younes nog doen? Orders die te bestellen zijn en
          # orders die "besteld" zijn maar waar geen Basetao-bevestiging aan hangt
          # ("ik ga het bestellen" gezegd, maar nooit echt gedaan — jij vergeet het
          # vaak, dus de bot blijft vragen tot het klopt).
          time.sleep(300)  # pas nadat alles opgestart is
          while True:
              try:
                  regels = []
                  with _ledlock:
                      d = ledger_load()
                      for o in d.get("orders", []):
                          st = o.get("order_status")
                          naam = o.get("customer") or "?"
                          items = o.get("items") or "?"
                          prijs = float(o.get("price_eur") or 0)
                          if st == "te_bestellen":
                              regels.append(f"❗ NOG BESTELLEN: #{o['num']} {naam} — {items} (€{prijs:g})")
                          elif st == "besteld" and not o.get("basetao_ids"):
                              ts = str(o.get("updated") or o.get("created") or "")
                              try:
                                  oud = time.time() - time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
                              except Exception:  # noqa: BLE001
                                  oud = 999999
                              if oud > 12 * 3600:
                                  regels.append(f"❓ #{o['num']} {naam} — echt al besteld in Basetao? "
                                                f"Ik zie nog geen bevestiging ({items}).")
                  if regels:
                      _stuur_via_bridge(WA_SELF_JID,
                                        "⏰ Even jou ding, vergeet deze niet:\n" + "\n".join(regels[:8]))
              except Exception as e:  # noqa: BLE001
                  print("herinneringen faalden:", e)
              time.sleep(4 * 3600)

      threading.Thread(target=_herinneringen, daemon=True).start()
  
    def _wa_periodic_backup():
        # ververs de sessie-backup elke 30 min zolang we verbonden zijn,
        # zodat een herstart nooit een verouderde (Bad MAC) sessie terugzet
        while True:
            time.sleep(1800)
            try:
                if WA_STATE.get("status") == "connected":
                    wa_backup_up()
            except Exception as e:  # noqa: BLE001
                print("wa periodic backup failed:", e)

    threading.Thread(target=_wa_periodic_backup, daemon=True).start()

SHOP = """<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YZ Shop — premium reps</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0b0f0e;color:#eef2f0;margin:0;padding:0 0 60px}
header{padding:34px 24px 10px;max-width:1080px;margin:0 auto}
.brand{font-size:30px;font-weight:800;letter-spacing:.02em}
.brand span{color:#f7941d}
.sub{color:#8fa39b;margin-top:6px;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px;
max-width:1080px;margin:26px auto 0;padding:0 24px}
.pcard{background:#141c1a;border:1px solid #22332c;border-radius:14px;padding:18px;
display:flex;flex-direction:column;gap:10px}
.ptag{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#f7941d}
.pcard h3{margin:0;font-size:16px;line-height:1.35;font-weight:650}
.pprice{font-size:20px;font-weight:800}
.pnote{color:#8fa39b;font-size:12px;margin-top:-4px}
.pbtn{margin-top:auto;background:#f7941d;color:#10130f;text-align:center;font-weight:800;
text-decoration:none;border-radius:10px;padding:11px;font-size:14px}
.pbtn:hover{filter:brightness(1.08)}
.empty{grid-column:1/-1;color:#8fa39b;background:#141c1a;border:1px dashed #22332c;
border-radius:14px;padding:30px;text-align:center}
footer{max-width:1080px;margin:44px auto 0;padding:0 24px;color:#5f7268;font-size:12px;
border-top:1px solid #1a2620;padding-top:18px}
footer a{color:#f7941d}
.tg{position:fixed;right:20px;bottom:20px;background:#2aabee;color:#fff;font-weight:800;
border-radius:999px;padding:13px 20px;text-decoration:none;box-shadow:0 6px 24px rgba(0,0,0,.45)}
</style></head><body>
<header>
<div class="brand">YZ<span> SHOP</span></div>
<div class="sub">premium reps · handgepickte drops · levering 2–3 weken · betaling bij ontvangst</div>
</header>
<div class="grid">
__CARDS__
</div>
<footer>YZ Shop · bestellen en betalen regelen we persoonlijk via
<a href="https://t.me/younesrepbot">Telegram</a> · leverage levering NL · geen voorraad = op aanvraag</footer>
<a class="tg" href="https://t.me/younesrepbot">Bestel via Telegram</a>
</body></html>"""

_STATUS_NL = {"besteld": "in bestelling", "onderweg": "onderweg naar NL",
              "binnen": "op voorraad", "verpakken": "klaar gemaakt",
              "klaar": "direct leverbaar", "te_bestellen": "bestel ik zo",
              "interesse": "op aanvraag", "prijs_gegeven": "op aanvraag"}

def render_shop():
    with _ledlock:
        d = ledger_load()
    items = [o for o in d.get("orders", [])
             if (o.get("customer") or "").strip().lower() in ("onbekend", "", "?")]
    cards = []
    for o in reversed(items[-24:]):
        price = ("€%g" % o["price_eur"]) if o.get("price_eur") else "Prijs in DM"
        st = _STATUS_NL.get(o.get("order_status", ""), o.get("order_status", ""))
        bt = ""
        if o.get("basetao_ids"):
            bt = ('<div class="pnote">QC: <a style="color:#f7941d" target="_blank" rel="noopener" '
                  'href="https://www.basetao.com/best-taobao-agent-service/purchase/order_img/%s.html">'
                  "foto's</a></div>" % o["basetao_ids"][0])
        cards.append(
            '<div class="pcard"><div class="ptag">%s</div><h3>%s</h3>'
            '<div class="pprice">%s</div>%s%s'
            '<a class="pbtn" href="https://t.me/younesrepbot">Bestel via Telegram</a></div>'
            % (st, o.get("items") or "Custom item", price, "", bt))
    if not cards:
        cards.append('<div class="empty">Nieuwe drop onderweg — DM voor de huidige voorraad 📦</div>')
    return (SHOP.replace("__CARDS__", "\n".join(cards))
                .replace("__YEAR__", time.strftime("%Y")))

SITE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "static", "index.html")

@app.get("/admin", response_class=HTMLResponse)
def admin():
    return render_dashboard()

@app.get("/healthz")
def healthz():
    return {"ok": True, "bot_configured": bool(TELEGRAM_TOKEN),
            "whatsapp": WA_STATE["status"] if WHATSAPP_ENABLED else "uit"}

@app.get("/stats")
def stats():
    with _ledlock:
        d = ledger_load()
    return JSONResponse({"stats": compute_stats(d),
                         "maanden": monthly_breakdown(d),
                         "recent": list(reversed(d["entries"][-25:]))})

@app.post("/api/entry")
async def api_entry(req: Request):
    b = await req.json()
    t, amt = b.get("type"), b.get("amount_eur")
    if t not in ("income", "cost", "topup") or not amt:
        return JSONResponse({"ok": False, "error": "type of bedrag ongeldig"}, status_code=400)
    standaard = {"income": "cash", "cost": "basetao", "topup": "ideal"}[t]
    with _ledlock:
        d = ledger_load()
        d["entries"].append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "type": t,
            "amount_eur": float(amt),
            "method": b.get("method") or standaard,
            "customer": b.get("customer"), "items": None,
            "note": b.get("note") or "website", "source": "website"})
        ledger_save(d)
        s = compute_stats(d)
        hf_sync_up()
    return JSONResponse({"ok": True, "stats": s})

@app.post("/api/note")
async def api_note(req: Request):
    """Vrije notitie in het kasboek (bijv. herstelde transcripts, memo's)."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    note = (b.get("note") or "").strip()
    if not note:
        return JSONResponse({"ok": False, "error": "geen notitie"}, status_code=400)
    with _ledlock:
        d = ledger_load()
        d["entries"].append({"ts": _now(), "type": "note", "note": note[:2000],
                             "source": b.get("source") or "website"})
        ledger_save(d)
        hf_sync_up()
    return JSONResponse({"ok": True})

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
        reply = await asyncio.to_thread(agent_reply, "audio", text)
    except Exception as e:  # noqa: BLE001
        print("agent error:", e)
        return JSONResponse({"ok": False, "transcript": text, "error": "verwerking mislukt"},
                            status_code=502)
    return JSONResponse({"ok": True, "transcript": text, "resultaat": reply})

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

@app.post("/api/agent")
async def api_agent(req: Request):
    """Zelfde agent- brein als Telegram, maar via HTTP (voor tests en de website)."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    text = (b.get("text") or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "geen tekst"}, status_code=400)
    answer = await asyncio.to_thread(agent_reply, "web", text)
    return JSONResponse({"ok": True, "resultaat": answer})

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

# ---------------- whatsapp (Baileys-bridge) ----------------
# De Node-bridge (whatsapp-bridge/bridge.js) koppelt het bestaande WhatsApp-nummer
# als "gekoppeld apparaat" (zoals WhatsApp Web) en post hier inkomende berichten.
# Typ Younes zelf iets in een chat (vanaf zijn telefoon), dan pauzeert de bot voor
# die chat zodat hij het gesprek handmatig kan overnemen. "/bot uit" = 7 dagen,
# "/bot aan" = weer inschakelen. Pauzes leven mee in ledger.json (overleeft restarts).
WA_STATE = {"qr": None, "qr_ts": 0.0, "status": "startend" if WHATSAPP_ENABLED else "uit",
            "error": "", "backup_ts": 0.0, "pair_code": None}

def _wa_key(jid):
    return str(jid or "").split("@")[0].split(":")[0]

def _wa_pause_map(d):
    paused = d.setdefault("whatsapp", {}).setdefault("paused", {})
    now = time.time()
    for k in [k for k, v in paused.items() if float(v or 0) < now]:
        paused.pop(k, None)
    return paused

def wa_is_paused(jid):
    with _ledlock:
        d = ledger_load()
    return _wa_key(jid) in _wa_pause_map(d)

def wa_pause(jid, hours=None, clear=False):
    with _ledlock:
        d = ledger_load()
        paused = _wa_pause_map(d)
        k = _wa_key(jid)
        if clear:
            paused.pop(k, None)
        else:
            paused[k] = time.time() + (hours if hours is not None else WA_TAKEOVER_HOURS) * 3600.0
        ledger_save(d)
        hf_sync_up()

def wa_backup_up():
    """Baileys-sessie meesturen naar de HF-dataset, zodat een redeploy geen
    nieuwe QR-scan vereist (sessie wordt bij het opstarten teruggezet)."""
    if not (HF_DATASET and HF_TOKEN):
        return
    try:
        import zipfile
        from huggingface_hub import HfApi
        zpath = os.path.join(os.environ.get("TMPDIR", "/tmp"), "wa-session.zip")
        if os.path.exists(zpath):
            os.remove(zpath)
        os.makedirs(WA_SESSION_DIR, exist_ok=True)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _dirs, files in os.walk(WA_SESSION_DIR):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, WA_SESSION_DIR))
        HfApi(token=HF_TOKEN).upload_file(
            path_or_fileobj=zpath, path_in_repo="wa-session.zip",
            repo_id=HF_DATASET, repo_type="dataset")
        print("wa-sessie gebackupt naar HF dataset")
    except Exception as e:  # noqa: BLE001
        print("wa backup up failed:", e)

def wa_backup_down():
    """Bij start: eerder gegpairde sessie uit de HF-dataset terugzetten (als de
    lokale sessie weg is, bijv. na een redeploy op een verse container)."""
    if not (HF_DATASET and HF_TOKEN):
        return
    if os.path.isdir(WA_SESSION_DIR) and os.listdir(WA_SESSION_DIR):
        return  # lokale sessie is leidend
    try:
        import zipfile
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                            filename="wa-session.zip", token=HF_TOKEN)
        os.makedirs(WA_SESSION_DIR, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            namen = z.namelist()
            z.extractall(WA_SESSION_DIR)
        print(f"wa-sessie hersteld uit HF dataset: {len(namen)} bestanden "
              f"(creds.json aanwezig: {'creds.json' in namen})")
    except Exception as e:  # noqa: BLE001
        print("wa backup down failed:", e)

WA_SELF_JID = "31684805378@s.whatsapp.net"  # 'bericht jezelf'-chat: notitie-bevestigingen landen hier

def _stuur_via_bridge(to, tekst):
    """Bericht direct via de bridge versturen (bericht-jezelf of een klant-chat)."""
    try:
        r = requests.post("http://127.0.0.1:7861/send",
                          json={"to": to, "text": tekst}, timeout=25)
        return r.ok
    except Exception as e:  # noqa: BLE001
        print("stuur via bridge faalde:", e)
        return False

def _bing_afbeeldingen(q, n=2):
    """Fallback productfoto's via Bing Images (als de doppel-bridge offline is).
    Geeft dataURL's terug die direct via WhatsApp verstuurd kunnen worden."""
    r = requests.get("https://www.bing.com/images/search", params={"q": q, "form": "HDRSC2"},
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"},
                     timeout=15)
    urls = re.findall(r'"murl":"(https?://[^"]+)"', r.text)
    out = []
    for u in urls:
        if len(out) >= n:
            break
        try:
            ir = requests.get(u.replace("\\u0026", "&"), timeout=12,
                              headers={"User-Agent": "Mozilla/5.0"})
            if ir.ok and len(ir.content) > 8000:
                out.append("data:image/jpeg;base64," + base64.b64encode(ir.content).decode())
        except Exception:  # noqa: BLE001
            continue
    return out

WA_WELCOME = """Yo, welkom bij YZ Shop

Stuur eerst je voornaam, zodat ik weet met wie ik app — WhatsApp toont soms alleen een nummer. Stuur daarna je Snapchatnaam, plus een foto of link van wat je zoekt en je maat.

Voetbalshirt met bedrukking: €30
Met broekje: €40

Stuur voor de bedrukking ook de naam en het rugnummer door. Ik app je zo terug!"""

@app.post("/whatsapp/incoming")
async def wa_incoming(req: Request):
    """Inkomend WhatsApp-bericht van de bridge -> antwoord (of None = zwijgen)."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    jid = str(b.get("chat") or "")
    if not jid:
        return JSONResponse({"ok": False, "error": "chat ontbreekt"}, status_code=400)
    key = _wa_key(jid)
    text = (b.get("text") or "").strip()
    mtype = b.get("type") or "text"

    # Laatste bericht + gespreksgeschiedenis per chat vastleggen (staat-van-zaken
    # + de bot kan teruglezen wat de klant al gezegd heeft, ook na herstart)
    try:
        with _ledlock:
            d = ledger_load()
            c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(key, {})
            if b.get("name"):
                c.setdefault("naam", str(b["name"])[:80])
            if jid:
                c["jid"] = jid  # volledige adres (kan een LID zijn) voor direct sturen
            c["laatste"] = {"ts": _now(), "van": "jij" if b.get("from_me") else "klant",
                            "tekst": (text[:200] or f"[{mtype}]")}
            if text:
                g = c.setdefault("gesprek", [])
                g.append({"r": "j" if b.get("from_me") else "k", "t": text[:200]})
                del g[:-30]
            ledger_save(d)
    except Exception:  # noqa: BLE001
        pass

    # Gespreksgeheugen herstellen uit het kasboek als het leeg is (na herstart/deploy):
    # zo stelt de bot nooit meer vragen die de klant al beantwoord heeft.
    ck = "wa:" + key
    if not _chatmem.get(ck):
        try:
            with _ledlock:
                d = ledger_load()
                c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(key, {})
                for g in (c.get("gesprek") or [])[-14:]:
                    rol = "user"
                    t = g.get("t", "")
                    if g.get("r") == "j":
                        t = "(jij zelf): " + t
                    _chatmem.setdefault(ck, []).append({"role": rol, "content": t})
        except Exception:  # noqa: BLE001
            pass

    if b.get("from_me"):
        low = text.lower()
        if low in ("/bot aan", "bot aan"):
            wa_pause(jid, clear=True)
            return JSONResponse({"reply": "🤖 Bot weer AAN in deze chat.", "to_me": True})
        if low in ("/bot uit", "bot uit"):
            wa_pause(jid, hours=24 * 7)
            return JSONResponse({"reply": "🤖 Bot UIT in deze chat (7 dagen). "
                                          "Typ 'bot aan' om weer in te schakelen.", "to_me": True})
        # Prijslijst vullen: "prijs alo runner 155" -> staat voortaan in vaste_prijzen
        if low.startswith("prijs "):
            rest = text[len("prijs "):].strip()
            m = re.search(r"(\d+(?:[.,]\d{1,2})?)", rest)
            if not m:
                return JSONResponse({"reply": "📝 Format: prijs <product> <bedrag> "
                                              "(bijv: prijs alo runner 155)",
                                     "to_me": True, "to_chat": WA_SELF_JID})
            bedrag = float(m.group(1).replace(",", "."))
            prod = (rest[:m.start()] + " " + rest[m.end():]).strip().lower()[:60] or "onbekend"
            with _ledlock:
                d = ledger_load()
                d.setdefault("whatsapp", {}).setdefault("vaste_prijzen", {})[prod] = bedrag
                ledger_save(d)
                hf_sync_up()
            return JSONResponse({"reply": f"✅ Staat in je prijslijst: {prod} = €{bedrag:g}",
                                 "to_me": True, "to_chat": WA_SELF_JID})
        # Openstaande klantvragen: "open vragen" -> lijst met nummers
        if low in ("open vragen", "vragen", "openstaande vragen"):
            with _ledlock:
                d = ledger_load()
            pend = [p for p in d.get("whatsapp", {}).get("pending", []) if not p.get("answered")]
            if not pend:
                txt = "Geen open vragen 👍"
            else:
                txt = "❓ Open vragen:\n" + "\n".join(
                    f"#{p['n']} {p['klant']}: {p['vraag']} [{str(p['ts'])[11:16]}]"
                    for p in pend[-10:])
            return JSONResponse({"reply": txt, "to_me": True, "to_chat": WA_SELF_JID})
        # Jouw antwoord op een doorgestuurde klantvraag: "antwoord 2 kost 130, stuur ik vanavond"
        if low.startswith("antwoord"):
            rest = text[len("antwoord"):].strip()
            mm = re.match(r"(\d+)\s+(.+)", rest, re.S)
            if not mm:
                return JSONResponse({"reply": "📝 Format: antwoord <nummer> <je antwoord> "
                                              "(nummers: stuur 'open vragen')",
                                     "to_me": True, "to_chat": WA_SELF_JID})
            n, antw = int(mm.group(1)), mm.group(2).strip()[:600]
            with _ledlock:
                d = ledger_load()
                pend = d.get("whatsapp", {}).get("pending", [])
                hit = next((p for p in reversed(pend)
                            if p.get("n") == n and not p.get("answered")), None)
                if not hit:
                    return JSONResponse({"reply": f"Vraag #{n} niet gevonden (al beantwoord?).",
                                         "to_me": True, "to_chat": WA_SELF_JID})
                hit["answered"] = True
                hit["antwoord"] = antw
                mp = re.search(r"€\s?(\d+(?:[.,]\d{1,2})?)", antw)
                bedrag = float(mp.group(1).replace(",", ".")) if mp else None
                # prijs-geheugen: prijzen die je een keer geeft, weet de bot voortaan zelf
                if bedrag and re.search(r"prijs|kosten|hoeveel", hit.get("vraag", ""), re.I):
                    vm = re.search(r"(?:prijs|kosten)\s*(?:van)?\s*[:\-]?\s*(.+)",
                                   hit.get("vraag", ""), re.I)
                    prod = (vm.group(1) if vm else hit.get("vraag", ""))[:60].strip(" ?.")
                    d.setdefault("whatsapp", {}).setdefault("prijzen", {})[prod.lower()] = bedrag
                c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(
                    hit["chat_key"][3:], {})
                if bedrag:
                    c["prijs_afgesproken"] = bedrag
                    c["prijs_gegeven"] = True
                target_jid = c.get("jid") or (hit["chat_key"][3:] + "@s.whatsapp.net")
                ledger_save(d)
                hf_sync_up()
            threading.Thread(target=_stuur_via_bridge,
                             args=(target_jid, antw),
                             daemon=True).start()
            return JSONResponse({"reply": f"✅ Doorgestuurd naar {hit['klant']}: {antw[:80]}",
                                 "to_me": True, "to_chat": WA_SELF_JID})
        # Dagoverzicht: "stand van vandaag" / "overzicht" -> per klant wat hij wil
        # en wat er nog mist, plus vandaag genoteerde bestellingen.
        if any(kw in low for kw in ("stand van", "overzicht", "wat is de stand", "dagoverzicht")):
            with _ledlock:
                d = ledger_load()
            chats = d.get("whatsapp", {}).get("chats", {})
            regels = []
            for num, c in sorted(chats.items(),
                                 key=lambda kv: str((kv[1].get("laatste") or {}).get("ts") or ""),
                                 reverse=True):
                naam = c.get("klantnaam") or c.get("naam") or ("+" + num)
                deel = "• " + naam
                if c.get("gezocht"):
                    deel += " — " + c["gezocht"]
                if c.get("ontbreekt") and c["ontbreekt"].lower() not in ("leeg", "niets", "niks", "-"):
                    deel += " (mist: " + c["ontbreekt"] + ")"
                laat = c.get("laatste") or {}
                if laat.get("ts"):
                    deel += " [" + str(laat["ts"])[11:16] + "]"
                regels.append(deel)
            vandaag = _now()[:10]
            bestellingen = [o for o in d.get("orders", [])
                            if str(o.get("created", "")).startswith(vandaag)]
            if bestellingen:
                regels.append("Vandaag besteld: " + ", ".join(
                    f"#{o['num']} {o['customer']} ({o.get('items') or '?'})"
                    for o in bestellingen[-8:]))
            for p in [p for p in d.get("whatsapp", {}).get("pending", [])
                      if not p.get("answered")][-8:]:
                regels.append(f"❓ {p['klant']}: {p['vraag']} → antwoord {p['n']} <antwoord>")
            if not regels:
                txt = "📋 Nog niks voor vandaag — geen open klant-chats."
            else:
                txt = "📋 Stand van vandaag:\n" + "\n".join(regels[:18])
            return JSONResponse({"reply": txt, "to_me": True, "to_chat": WA_SELF_JID})
        # Jouw eigen notities vanaf je telefoon: "besteld <klant> <wat> (€bedrag)" of
        # "betaald <klant> (€bedrag)" — zo weet de bot wat je al hebt besteld/ontvangen.
        if low.startswith("besteld"):
            rest = text[len("besteld"):].strip()
            delen = rest.split()
            if not delen:
                return JSONResponse({"reply": "📝 Format: besteld <klant> <wat> (€bedrag)", "to_me": True})
            naam = delen[0]
            itemstxt = " ".join(delen[1:]) or "?"
            m = re.search(r"(?:€|eur\s?)\s?(\d+(?:[.,]\d{1,2})?)", itemstxt, re.I)
            prijs = float(m.group(1).replace(",", ".")) if m else 0
            if m:
                itemstxt = (itemstxt[:m.start()] + " " + itemstxt[m.end():]).strip() or "?"
            out = str(run_tool("create_order", json.dumps(
                {"customer": naam, "items": itemstxt, "price_eur": prijs})))
            mm = re.search(r"#(\d+)", out)
            if mm:
                out = str(run_tool("update_order", json.dumps(
                    {"num": int(mm.group(1)), "order_status": "besteld"}))) or out
            # bevestiging naar je eigen 'bericht jezelf'-chat, niet in de klant z'n chat
            return JSONResponse({"reply": "📝 " + out, "to_me": True, "to_chat": WA_SELF_JID})
        if low.startswith("betaald"):
            rest = text[len("betaald"):].strip()
            naam = rest.split()[0] if rest.split() else ""
            m = re.search(r"(?:€|eur\s?)\s?(\d+(?:[.,]\d{1,2})?)", rest, re.I)
            with _ledlock:
                d = ledger_load()
                hit = None
                for o in reversed(d.get("orders", [])):
                    if naam and naam.lower() in str(o.get("customer", "")).lower() \
                            and o.get("payment_status") != "betaald":
                        hit = o
                        break
                if hit:
                    hit["payment_status"] = "betaald"
                    hit.setdefault("history", []).append({"ts": _now(), "event": "betaald (notitie in WhatsApp)"})
                    d.setdefault("entries", []).append({
                        "ts": _now(), "type": "income",
                        "amount_eur": float(m.group(1).replace(",", ".")) if m else float(hit.get("price_eur") or 0),
                        "method": "bank", "customer": hit.get("customer"),
                        "note": "notitie in WhatsApp"})
                    ledger_save(d)
                    hf_sync_up()
                    return JSONResponse({"reply": f"✅ Order #{hit['num']} ({hit['customer']}) op betaald gezet "
                                                  f"en geld bijgeschreven.", "to_me": True,
                                         "to_chat": WA_SELF_JID})
            return JSONResponse({"reply": f"❓ Geen open order gevonden voor '{naam}'. "
                                          f"Check de klantnaam (zelfde spelling als in de order).",
                                 "to_me": True, "to_chat": WA_SELF_JID})
        if WA_TAKEOVER_HOURS > 0:
            wa_pause(jid)  # human takeover: jij hebt zelf geantwoord
            return JSONResponse({"reply": None, "note": f"pauze {WA_TAKEOVER_HOURS:g}u"})
        return JSONResponse({"reply": None, "note": "takeover uit — bot blijft actief"})

    if wa_is_paused(jid):
        return JSONResponse({"reply": None, "note": "pauze"})

    # Eerste bericht van dit nummer ooit -> welkomstbericht van Younes (1x per nummer,
    # staat in de ledger dus dit overleeft restarts/redeploys).
    with _ledlock:
        d = ledger_load()
        welcomed = d.setdefault("whatsapp", {}).setdefault("welcomed", {})
        is_new = key not in welcomed
        if is_new:
            welcomed[key] = True
            ledger_save(d)
    if is_new:
        threading.Thread(target=hf_sync_up, daemon=True).start()
        return JSONResponse({"reply": WA_WELCOME})

    if mtype == "audio":
        b64 = b.get("audio_b64") or ""
        if not b64:
            return JSONResponse({"reply": None})
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wa_{int(time.time() * 1000)}.ogg")
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        audio_bytes = b""
        try:
            text = await asyncio.to_thread(transcribe, path)
            with open(path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:  # noqa: BLE001
            print("wa asr error:", e)
            return JSONResponse({"reply": "Stemmetje kon ik niet verwerken 🙈 typ even wat je zoekt."})
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        if not text:
            return JSONResponse({"reply": "❓ Geen spraak herkend, typ even."})
        voice_note_opslaan(text, "whatsapp", chat_key="wa:" + key, audio_bytes=audio_bytes)
    elif mtype == "image":
        # Foto onthouden in de chat-status: de bot mag daarna NIET meer om een
        # foto vragen (klantklacht: "hij zegt steeds dat er geen foto's zijn").
        try:
            with _ledlock:
                d = ledger_load()
                c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(key, {})
                c["foto_ontvangen"] = True
                ledger_save(d)
        except Exception:  # noqa: BLE001
            pass
        if not text:
            hist = _chatmem.setdefault("wa:" + key, [])
            hist.append({"role": "user", "content": "[klant stuurde een foto, zonder tekst]"})
            del hist[:-40]
            return JSONResponse({"reply": WA_IMG_REPLY})
        text = "[foto] " + text

    if not text:
        return JSONResponse({"reply": None})
    try:
        answer = await asyncio.to_thread(agent_reply, "wa:" + key, text)
    except Exception as e:  # noqa: BLE001
        print("wa agent error:", e)
        return JSONResponse({"reply": None})  # stil falen; Younes ziet de chat in de app
    try:  # bot-antwoorden ook in het opgeslagen gesprek (teruglezen na herstart)
        with _ledlock:
            d = ledger_load()
            c = d.setdefault("whatsapp", {}).setdefault("chats", {}).setdefault(key, {})
            g = c.setdefault("gesprek", [])
            g.append({"r": "b", "t": str(answer)[:200]})
            del g[:-30]
            ledger_save(d)
    except Exception:  # noqa: BLE001
        pass

    # Mini-briefing naar Younes: wie appt, wat wil die, wat er nog mist.
    def _briefing():
        try:
            with _ledlock:
                d = ledger_load()
                c = d.get("whatsapp", {}).get("chats", {}).get(key, {})
            naam = c.get("klantnaam") or c.get("naam") or ("+" + key)
            zoek = c.get("gezocht") or "onbekend"
            if c.get("prijs_afgesproken"):
                prijs = "€" + str(c["prijs_afgesproken"])
            elif c.get("prijs_gegeven"):
                prijs = "prijs gegeven"
            else:
                prijs = "nog geen prijs"
            mist = c.get("ontbreekt") or "niks meer"
            _stuur_via_bridge(
                WA_SELF_JID,
                f"📥 {naam}: \"{text[:80]}\"\n👀 zoekt: {zoek} · 💶 {prijs} · ⚠️ mist: {mist}")
        except Exception as e:  # noqa: BLE001
            print("briefing faalde:", e)
    threading.Thread(target=_briefing, daemon=True).start()
    resp = {"reply": answer}
    with _srclock:
        imgs = _wa_pending_images.pop("wa:" + key, None)
    if imgs:
        resp["images"] = [i.split(",", 1)[-1] for i in imgs]  # dataURL -> ruwe base64
    return JSONResponse(resp)

@app.post("/whatsapp/history_chats")
async def wa_history_chats(req: Request):
    """Bridge stuurt de chat-jids uit de WhatsApp-historie (iedereen die ooit
    appte) -> opslaan als broadcast-doelen."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    nieuw = 0
    with _ledlock:
        d = ledger_load()
        known = d.setdefault("whatsapp", {}).setdefault("known_chats", {})
        for jid in (b.get("chats") or [])[:800]:
            if isinstance(jid, str) and jid.endswith("@s.whatsapp.net"):
                k = _wa_key(jid)
                if k and k not in known:
                    known[k] = {"jid": jid}
                    nieuw += 1
        ledger_save(d)
    print(f"historie-chats opgeslagen: {nieuw} nieuw")
    if nieuw:
        threading.Thread(target=hf_sync_up, daemon=True).start()
    return JSONResponse({"ok": True, "nieuw": nieuw})

@app.get("/whatsapp/contacts")
async def wa_contacts(req: Request):
    """Alle bekende chats (CRM + historie) met hun volledige adres."""
    key = req.query_params.get("key", "")
    if ACCESS_CODE and key != ACCESS_CODE:
        return JSONResponse({"ok": False, "error": "geen toegang"}, status_code=401)
    with _ledlock:
        d = ledger_load()
    chats = d.get("whatsapp", {}).get("chats", {})
    known = d.get("whatsapp", {}).get("known_chats", {})
    merged = {}
    for k, c in chats.items():
        merged[k] = {"key": k, "jid": (c.get("jid") or k + "@s.whatsapp.net"),
                     "naam": (c.get("klantnaam") or c.get("naam") or "")}
    for k, c in known.items():
        if k not in merged:
            merged[k] = {"key": k, "jid": (c.get("jid") or k + "@s.whatsapp.net"), "naam": ""}
    return JSONResponse({"ok": True, "chats": list(merged.values())})

@app.post("/whatsapp/send")
async def wa_send(req: Request):
    """Direct een WhatsApp-bericht versturen via de bridge (test/debug).
    key in query is verplicht. Body: {"to": "<nummer|jid>", "text": "..."}"""
    key = req.query_params.get("key", "")
    if ACCESS_CODE and key != ACCESS_CODE:
        return JSONResponse({"ok": False, "error": "geen toegang"}, status_code=401)
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": "ongeldige json"}, status_code=400)
    to = str(b.get("to") or "").strip()
    tekst = str(b.get("text") or "").strip()
    if not to or not tekst:
        return JSONResponse({"ok": False, "error": "to en text verplicht"}, status_code=400)
    ok = await asyncio.to_thread(_stuur_via_bridge, to, tekst)
    return JSONResponse({"ok": ok})

@app.post("/whatsapp/notify_unreadable")
async def wa_notify_unreadable(req: Request):
    """Bridge kan een inkomend bericht niet ontsleutelen (CIPHERTEXT-stub, oude
    sessie na herkoppelen) -> waarschuw Younes in zijn bericht-jezelf-chat."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    num = str(b.get("num") or _wa_key(b.get("chat") or "")).strip()
    return JSONResponse({
        "reply": ("⚠️ Bericht van +" + num + " kon ik niet lezen (oude versleuteling na "
                  "herkoppelen). De chat ziet het wél op je telefoon. Vraag de afzender "
                  "het gesprek met je zakelijke nummer te verwijderen en opnieuw te "
                  "sturen — daarna kan de bot het wel lezen."),
        "to_me": True, "to_chat": WA_SELF_JID})


@app.post("/whatsapp/qr")
async def wa_qr_post(req: Request):
    """Bridge post hier elke nieuwe QR (data-URL) en/of de koppelcode."""
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    if "qr" in b:
        WA_STATE["qr"] = b.get("qr") or None
        WA_STATE["qr_ts"] = time.time()
        WA_STATE["status"] = "wacht op scan"
    if b.get("pair_code"):
        WA_STATE["pair_code"] = str(b["pair_code"])[:12]
        if WA_STATE["status"] in ("startend", "verbinden"):
            WA_STATE["status"] = "wacht op scan"
    return JSONResponse({"ok": True})

@app.get("/whatsapp/qr")
async def wa_qr_get():
    return JSONResponse({"status": WA_STATE["status"], "qr": WA_STATE["qr"],
                         "qr_age": int(time.time() - WA_STATE["qr_ts"]) if WA_STATE["qr"] else None,
                         "pair_code": WA_STATE["pair_code"],
                         "error": WA_STATE["error"], "enabled": WHATSAPP_ENABLED})

@app.post("/whatsapp/status")
async def wa_status_post(req: Request):
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    st = str(b.get("status") or "")
    if st:
        WA_STATE["status"] = st
    WA_STATE["error"] = str(b.get("error") or "")
    if st == "connected":
        WA_STATE["qr"] = None
        if (HF_DATASET and HF_TOKEN and time.time() - WA_STATE["backup_ts"] > 60):
            WA_STATE["backup_ts"] = time.time()

            def _bk():
                time.sleep(130)  # pas backuppen als de verbinding stabiel is (geen halve sessie)
                wa_backup_up()
            threading.Thread(target=_bk, daemon=True).start()
    return JSONResponse({"ok": True})

@app.get("/whatsapp/status")
async def wa_status_get():
    with _ledlock:
        d = ledger_load()
    return JSONResponse({"status": WA_STATE["status"], "enabled": WHATSAPP_ENABLED,
                         "error": WA_STATE["error"],
                         "gepauzeerde_chats": len(_wa_pause_map(d)),
                         "takeover_uren": WA_TAKEOVER_HOURS})

@app.post("/whatsapp/logout")
async def wa_logout_post():
    """Apparaat is op de telefoon afgemeld: de (nu dode) sessie ook uit de
    HF-backup wissen, zodat een herstart niet per ongeluk de oude sessie
    terugzet en de bridge in een 401-loop blijft hangen."""
    WA_STATE["status"] = "uitgelogd"
    WA_STATE["pair_code"] = None
    WA_STATE["qr"] = None
    if HF_DATASET and HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            HfApi(token=HF_TOKEN).delete_file(
                path_in_repo="wa-session.zip", repo_id=HF_DATASET, repo_type="dataset")
            print("dode wa-sessie uit HF dataset verwijderd")
        except Exception as e:  # noqa: BLE001
            print("wa logout cleanup failed:", e)
    return JSONResponse({"ok": True})

WA_CHATS_PAGE = """<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Stand van zaken — WhatsApp</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0f1716;color:#e8efec;margin:0;padding:20px;max-width:760px}
h1{font-size:20px}.sub{color:#8aa39c;font-size:13px;margin-bottom:14px}
.card{background:#182522;border:1px solid #24382f;border-radius:12px;padding:14px;margin-top:12px}
.naam{font-weight:700;font-size:16px}.badge{display:inline-block;background:#25d366;color:#06281a;
border-radius:20px;padding:2px 10px;font-size:12px;font-weight:700;margin-left:8px}
.badge.grijs{background:#31473f;color:#b9cdc6}
.rij{font-size:14px;margin-top:6px;color:#c8d6d1}.rij b{color:#e8efec}
.leeg{color:#8aa39c}.laatste{color:#8aa39c;font-size:13px;margin-top:8px;border-top:1px solid #24382f;padding-top:8px}
.ontbreekt{color:#ffb86b;font-weight:600}
.banner{border-radius:12px;padding:12px 14px;margin-bottom:6px;font-weight:700;font-size:15px}
.banner.ok{background:#123a28;border:1px solid #25d366;color:#7dffb8}
.banner.slecht{background:#3a1212;border:1px solid #ff6b6b;color:#ffb0b0}
.banner .subfont{display:block;font-weight:400;font-size:13px;margin-top:4px;color:inherit;opacity:.85}
</style></head><body>
<h1>📋 Stand van zaken</h1>
<div class="sub">Elke klant-chat + wat er nog mist · ververst elke 30s · besteld/betaald noteren kan gewoon in WhatsApp: "besteld haroun groene dunks €110"</div>
__BANNER__
__CONTENT__
</body></html>"""

@app.get("/wa-chats", response_class=HTMLResponse, include_in_schema=False)
def wa_chats_page(request: Request):
    key = request.query_params.get("key", "")
    if ACCESS_CODE and key != ACCESS_CODE:
        return HTMLResponse("<h1>401 — toegangscode ontbreekt</h1><p>voeg ?key=... toe aan de URL</p>",
                            status_code=401)
    with _ledlock:
        d = ledger_load()
    chats = d.get("whatsapp", {}).get("chats", {})
    paused = _wa_pause_map(dict(d))
    orders = d.get("orders", [])
    # Status-balk: is de bot nu actief en wanneer kwam het laatste bericht binnen?
    verbonden = WA_STATE.get("status") == "connected"
    laatste_ts = max([str((c.get("laatste") or {}).get("ts") or "")
                      for c in chats.values()] or [""])
    try:
        laatste_min = int((time.time() - datetime.fromisoformat(laatste_ts).timestamp()) / 60)
    except Exception:  # noqa: BLE001
        laatste_min = None
    if laatste_min is None:
        laatste_txt = "nog geen berichten ontvangen"
    elif laatste_min < 1:
        laatste_txt = "laatste bericht: net nu"
    elif laatste_min < 60:
        laatste_txt = f"laatste bericht: {laatste_min} min geleden"
    else:
        laatste_txt = f"laatste bericht: {laatste_min // 60} uur geleden"
    if verbonden:
        banner = ('<div class="banner ok">✅ Bot is ACTIEF en verbonden met WhatsApp'
                  f'<span class="subfont">{laatste_txt} · check: {datetime.now().strftime("%H:%M")}</span></div>')
    else:
        banner = (f'<div class="banner slecht">❌ Bot is NIET verbonden (status: {WA_STATE.get("status")})'
                  '<span class="subfont">Herstart de service of koppel opnieuw via /wa-qr</span></div>')
    if not chats:
        content = '<div class="card leeg">Nog geen chats binnengekomen.</div>'
    else:
        def _sort(kv):
            return str((kv[1].get("laatste") or {}).get("ts") or "")
        delen = []
        for num, c in sorted(chats.items(), key=_sort, reverse=True):
            naam = c.get("klantnaam") or c.get("naam") or ("+" + num)
            status = c.get("status") or "onbekend"
            badge = f'<span class="badge">{status}</span>' if num not in paused \
                else '<span class="badge grijs">bot uit</span>'
            rijen = []
            if c.get("gezocht"):
                rijen.append(f'<div class="rij">🔎 <b>Zoekt:</b> {c["gezocht"]}</div>')
            if c.get("maat"):
                rijen.append(f'<div class="rij">📏 <b>Maat:</b> {c["maat"]}</div>')
            if c.get("prijs_afgesproken") is not None:
                rijen.append(f'<div class="rij">💶 <b>Afgesproken:</b> €{float(c["prijs_afgesproken"]):g}</div>')
            elif c.get("prijs_gegeven"):
                rijen.append('<div class="rij">💶 prijs genoemd (bedrag niet vastgelegd)</div>')
            else:
                rijen.append('<div class="rij ontbreekt">💶 nog geen prijs genoemd</div>')
            rijen.append(f'<div class="rij">📷 foto/link: {"✅" if c.get("foto_ontvangen") else "❌ nog geen"}</div>')
            if c.get("ontbreekt") and c.get("ontbreekt").lower() not in ("leeg", "niets", "niks", "-"):
                rijen.append(f'<div class="rij ontbreekt">⚠️ Mist nog: {c["ontbreekt"]}</div>')
            matches = [o for o in orders
                       if (c.get("klantnaam") or "").lower()
                       and c["klantnaam"].lower() in str(o.get("customer", "")).lower()][-3:]
            if matches:
                otxt = " · ".join(f"#{o['num']} {o.get('items') or '?'} €{float(o.get('price_eur') or 0):g} "
                                  f"📦{o.get('order_status')} 💶{o.get('payment_status')}" for o in matches)
                rijen.append(f'<div class="rij">🛒 <b>Orders:</b> {otxt}</div>')
            laat = c.get("laatste") or {}
            laatste = f'{laat.get("van", "?")}: {laat.get("tekst", "")}' if laat else "nog niks"
            delen.append(
                f'<div class="card"><div class="naam">{naam} <span style="color:#8aa39c;font-size:12px">+{num}</span>{badge}</div>'
                + "".join(rijen)
                + f'<div class="laatste">💬 {laatste}</div></div>')
        content = "".join(delen)
    return HTMLResponse(WA_CHATS_PAGE.replace("__BANNER__", banner).replace("__CONTENT__", content))

WA_QR_PAGE = """<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WhatsApp koppelen — Rep Agent</title><style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0f1716;color:#e8efec;margin:0;padding:24px;max-width:560px}
h1{font-size:20px}.sub{color:#8aa39c;font-size:13px}
.card{background:#182522;border:1px solid #24382f;border-radius:12px;padding:18px;margin-top:16px}
input{background:#0c1210;border:1px solid #2b463c;color:#e8efec;border-radius:8px;padding:10px;width:100%;box-sizing:border-box}
button{background:#25d366;border:0;border-radius:8px;padding:10px 16px;font-weight:700;cursor:pointer;margin-top:10px}
#qr{max-width:320px;width:100%;background:#fff;border-radius:12px;padding:10px;margin-top:12px}
#stat{font-size:14px;margin-top:10px}
ol{color:#c8d6d1;font-size:14px;line-height:1.6}
.note{color:#8aa39c;font-size:12px;margin-top:14px}
</style></head><body>
<h1>📱 WhatsApp koppelen</h1>
<div class="sub">Koppel je bestaande nummer als apparaat — je telefoon blijft gewoon werken.</div>
<div class="card"><input id="key" placeholder="toegangscode"><button onclick="save()">Opslaan</button>
<div id="stat">laden…</div><img id="qr" style="display:none" alt="QR-code">
<div id="pair" style="display:none">
<div style="margin-top:16px;color:#8aa39c;font-size:13px">Geen tweede scherm? Koppel <b>met code</b>:</div>
<div style="font-size:34px;font-weight:800;letter-spacing:.1em;margin:6px 0" id="paircode"></div>
<div class="note">WhatsApp → <b>Instellingen → Gekoppelde apparaten → Apparaat koppelen</b> →
onderaan <b>'Koppelen met telefoonnummer in plaats daarvan'</b> → typ deze code.<br>
(De QR hierboven werkt alleen vanaf een ánder scherm, bijv. je laptop.)</div></div>
<ol style="margin-top:16px"><li>Open <b>WhatsApp</b> op je telefoon</li><li><b>Instellingen → Gekoppelde apparaten → Apparaat koppelen</b></li>
<li>Scan de QR <i>vanaf een ander scherm</i>, of gebruik de code hierboven</li></ol>
<div class="note">Na het koppelen blijft alles gewoon zichtbaar op je telefoon.<br>
__TAKEOVER_NOTE__
Typ <b>bot uit</b> in een chat = bot 7 dagen uit · <b>bot aan</b> = weer aan.</div>
</div>
<script>let K=new URLSearchParams(location.search).get('key')||(document.cookie.split('; ').find(r=>r.startsWith('key='))||'').slice(4)||'';
if(K)document.getElementById('key').value=K;
function save(){K=document.getElementById('key').value.trim();document.cookie='key='+K+';path=/;max-age=31536000';tick();}
async function tick(){try{const r=await fetch('/whatsapp/qr',{headers:{'X-Access-Code':K}});
if(r.status===401){document.getElementById('stat').textContent='Eerst je toegangscode invullen.';document.getElementById('qr').style.display='none';return;}
const d=await r.json();const s=document.getElementById('stat');const q=document.getElementById('qr');const p=document.getElementById('pair');
if(d.pair_code){p.style.display='block';document.getElementById('paircode').textContent=d.pair_code;}else{p.style.display='none';}
if(d.status==='connected'){s.textContent='✅ WhatsApp is verbonden! Je kunt dit tabblad sluiten.';q.style.display='none';p.style.display='none';}
else if(d.qr&&String(d.qr).startsWith('data:image')){s.textContent='⏳ Wachten op scan… (QR '+d.qr_age+'s oud)';q.src=d.qr;q.style.display='block';}
else if(d.status==='uitgelogd'){s.textContent='Sessie verlopen — herstart de service en koppel opnieuw.';q.style.display='none';}
else{s.textContent='⏳ Status: '+d.status+' — QR verschijnt vanzelf…';q.style.display='none';}}catch(e){document.getElementById('stat').textContent='geen verbinding met de bridge…';}}
setInterval(tick,5000);tick();</script></body></html>"""

@app.get("/wa-qr", response_class=HTMLResponse, include_in_schema=False)
def wa_qr_page():
    if WA_TAKEOVER_HOURS > 0:
        note = f"Reageer jij zelf in een chat? Dan stopt de bot daar voor {WA_TAKEOVER_HOURS:g} uur.<br>"
    else:
        note = "Jij en de bot kunnen allebei in een chat reageren — de bot gaat niet uit als jij typt.<br>"
    return HTMLResponse(WA_QR_PAGE.replace("__TAKEOVER_NOTE__", note))

# ---------------- sourcing via doppel-bridge (userscript) ----------------
# De Tampermonkey-userscript (doppel-bridge.user.js) draait in Younes' Chrome op
# doppel.fit en voert zoekopdrachten uit: /s?query=... -> top items -> QC-foto's.
# Resultaten (incl. eventueel ge-cropte foto's zonder bovenwatermerk) komen hier
# terug en worden door de zoek_qc-tool als antwoord aan de klant gegeven.
SOURCE_JOBS = {}
SOURCE_RESULTS = {}
_srclock = threading.Lock()
_wa_pending_images = {}

@app.get("/whatsapp/source/next")
def source_next():
    """Userscript pollt hier: oudste wachtende zoekopdracht ophalen."""
    with _srclock:
        for jid, job in SOURCE_JOBS.items():
            if not job.get("dispatched"):
                job["dispatched"] = True
                return JSONResponse({"job": job})
    return JSONResponse({"job": None})

@app.post("/whatsapp/source/result")
async def source_result(req: Request):
    try:
        b = await req.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False}, status_code=400)
    jid = str(b.get("id") or "")
    if not jid:
        return JSONResponse({"ok": False, "error": "id ontbreekt"}, status_code=400)
    with _srclock:
        SOURCE_RESULTS[jid] = {"items": b.get("items") or [], "error": b.get("error"),
                               "ts": time.time()}
        # oude resultaten (1 uur) en jobs (10 min) opruimen
        nu = time.time()
        for k in [k for k, v in SOURCE_RESULTS.items() if nu - v.get("ts", nu) > 3600]:
            SOURCE_RESULTS.pop(k, None)
        for k in [k for k, v in SOURCE_JOBS.items() if nu - v.get("created_ts", nu) > 600]:
            SOURCE_JOBS.pop(k, None)
    return JSONResponse({"ok": True})

def tool_zoek_qc(query, count=3, chat_key=None):
    """Zoekopdracht naar de doppel-bridge sturen en op het resultaat wachten."""
    q = (query or "").strip()
    if not q:
        return "geef een zoekopdracht op"
    try:
        count = max(1, min(int(count or 3), 5))
    except (TypeError, ValueError):
        count = 3
    import urllib.parse
    import uuid
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "url": "https://doppel.fit/s?query=" + urllib.parse.quote(q),
           "count": count, "created": _now(), "created_ts": time.time(), "dispatched": False}
    with _srclock:
        SOURCE_JOBS[jid] = job
    deadline = time.time() + 75
    res = None
    while time.time() < deadline:
        with _srclock:
            res = SOURCE_RESULTS.pop(jid, None)
        if res:
            break
        time.sleep(3)
    if not res:
        if chat_key and str(chat_key).startswith("wa:"):
            # doppel offline -> productfoto's via Bing als fallback naar de klant
            try:
                urls = _bing_afbeeldingen(q, 2)
            except Exception:  # noqa: BLE001
                urls = []
            if urls:
                with _srclock:
                    _wa_pending_images[chat_key] = urls
                return ("Bijgaand referentiefoto's van '" + q + "'. DIT zijn geen prijzen — "
                        "noem er nooit bedragen bij. Zeg: 'dit bedoel je toch? De echte QC-foto's "
                        "stuur ik zo' en vraag ondertussen naar de maat.")
            return ("Zeg kort: 'ik pak de foto's even op, ik stuur ze zo 👍' en vraag "
                    "ondertussen naar de maat. (intern: doppel-bridge offline, geen bing-resultaat)")
        return ("doppel-bridge reageert niet — staat je Chrome open met doppel.fit "
                "en de Rep Agent userscript aan?")
    if res.get("error") or not res.get("items"):
        return "doppel gaf geen resultaten" + (f" ({res.get('error')})" if res.get("error") else "")
    items = [it for it in res["items"] if isinstance(it, dict)]
    # foto's klaarzetten voor WhatsApp (gecropte b64 zonder bovenwatermerk)
    if chat_key:
        imgs = [it.get("foto_b64") for it in items if it.get("foto_b64")]
        imgs = [i for i in imgs if isinstance(i, str) and i.startswith("data:image")]
        if imgs and chat_key.startswith("wa:"):
            _wa_pending_images[chat_key] = imgs[:2]
    lines = []
    for i, it in enumerate(items, 1):
        regel = f"{i}. {(it.get('titel') or '?')[:70]}"
        if it.get("prijs"):
            regel += f" — {it['prijs']}"
        if it.get("verkoper"):
            regel += f" | {str(it['verkoper'])[:40]}"
        lines.append(regel)
        if it.get("url"):
            lines.append(f"   {it['url']}")
        for f in (it.get("fotos") or [])[:2]:
            lines.append(f"   QC: {f}")
    return f"Top {len(items)} op doppel.fit voor '{q}':\n" + "\n".join(lines)

# ---------------- checkout via basetao wallet ----------------
BASETAO_BASE = "https://www.basetao.com/best-taobao-agent-service"
BASETAO_PAY_METHOD = os.environ.get("BASETAO_PAY_METHOD", "ideal").strip()

def _bt_session():
    if not BASETAO_COOKIE:
        raise RuntimeError("BASETAO_COOKIE niet ingesteld (Render secret)")
    s = requests.Session()
    s.headers.update({
        "Cookie": BASETAO_COOKIE,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
        "Referer": BASETAO_BASE + "/my_account/account/recharge.html",
        "X-Requested-With": "XMLHttpRequest",
    })
    return s

def basetao_create_recharge(amount_eur: float):
    """Maak een basetao top-up order voor exact dit bedrag en geef de gehoste
    betaalpagina-URL terug: een normale iDEAL-checkout voor de klant, het
    bedrag komt op onze eigen basetao-wallet."""
    s = _bt_session()
    r = s.get(BASETAO_BASE + "/my_account/account/recharge.html", timeout=30)
    m = re.search(r'name="bt_sb_token"[^>]*value="([0-9a-f]+)"', r.text)
    if not m:
        raise RuntimeError("basetao-token niet gevonden (cookie verlopen?)")
    r2 = s.post(BASETAO_BASE + "/account/recharge",
                data={"bt_sb_token": m.group(1),
                      "data": json.dumps({"option": BASETAO_PAY_METHOD,
                                          "money": f"{amount_eur:.2f}"})},
                timeout=30)
    try:
        out = r2.json()
    except ValueError:
        raise RuntimeError("basetao gaf geen JSON (Cloudflare?) http " + str(r2.status_code))
    if str(out.get("value")) != "1":
        raise RuntimeError("basetao weigerde: " + str(out.get("msg"))[:120])
    return BASETAO_BASE + "/torecharge/index/" + str(out["msg"]) + ".html"

def _checkout_error(status: int, bericht: str):
    """Vriendelijke HTML-foutpagina voor /checkout i.p.v. kale JSON."""
    html = (
        '<!doctype html><html lang="nl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Betaling niet mogelijk — YZ Shop</title><style>'
        'body{font-family:system-ui,Segoe UI,sans-serif;background:#0c0d10;color:#f2f3f5;'
        'margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}'
        '.card{max-width:460px;text-align:center}'
        'h1{font-size:22px;font-weight:700;margin:0 0 12px}'
        'p{color:#9aa1ad;font-size:15px;line-height:1.6;margin:0 0 28px}'
        'a{display:inline-block;color:#f7941d;font-weight:700;text-decoration:none;font-size:15px;'
        'border:1px solid #2a2d33;border-radius:10px;padding:10px 18px}'
        'a:hover{filter:brightness(1.15)}'
        '</style></head><body><div class="card">'
        '<h1>Betaling niet mogelijk</h1>'
        '<p>' + bericht + '<br>Probeer het over een paar minuten opnieuw '
        'of neem contact op via WhatsApp.</p>'
        '<a href="/">&larr; Terug naar de shop</a>'
        '</div></body></html>')
    return HTMLResponse(html, status_code=status)

@app.get("/checkout", include_in_schema=False)
def checkout(amount: str = ""):
    try:
        amt = round(float(str(amount).replace(",", ".")), 2)
    except ValueError:
        return _checkout_error(400, "Het opgegeven bedrag is ongeldig.")
    if amt < 10:
        return _checkout_error(400, "De minimale betaling is €10.")
    if amt > 5000:
        return _checkout_error(400, "De maximale betaling is €5000.")
    try:
        pay_url = basetao_create_recharge(amt)
    except Exception as e:  # noqa: BLE001
        print("checkout error:", e)
        return _checkout_error(502, "De betaalpartner accepteert de betaling nu even niet.")
    # topup meteen boeken: zo is "gestort via links" altijd vergelijkbaar met de kosten
    with _ledlock:
        d = ledger_load()
        d["entries"].append({
            "ts": _now(), "type": "topup", "amount_eur": amt, "method": "ideal",
            "note": "checkout-link aangemaakt (iDEAL → basetao wallet)", "source": "checkout"})
        ledger_save(d)
    threading.Thread(target=hf_sync_up, daemon=True).start()
    return RedirectResponse(pay_url, status_code=302)

_IMMUTABLE_EXTS = {".webp", ".png", ".jpg", ".jpeg", ".svg", ".ico",
                   ".woff", ".woff2", ".css", ".js"}

@app.get("/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
def page_fallback(full_path: str):
    """Volledige statische handler voor de gespiegelde site (incl. .html-mapping)."""
    if ".." in full_path:
        raise HTTPException(status_code=404)
    base = os.path.join("static", full_path.replace("/", os.sep))
    if full_path == "" or os.path.isdir(base):
        cand = os.path.join(base if os.path.isdir(base) else "static", "index.html")
    elif os.path.isfile(base):
        cand = base
    elif os.path.isfile(base + ".html"):
        cand = base + ".html"
    else:
        raise HTTPException(status_code=404)
    if not os.path.isfile(cand):  # map zonder index.html -> 404 i.p.v. 500
        raise HTTPException(status_code=404)
    mime, _ = mimetypes.guess_type(cand)
    if mime is None and cand.endswith(".webp"):
        mime = "image/webp"
    ext = os.path.splitext(cand)[1].lower()
    if ext in _IMMUTABLE_EXTS:
        cache_hdr = "public, max-age=31536000, immutable"  # content-gehashde assets
    else:
        cache_hdr = "no-cache, max-age=0"  # .html en bestanden zonder extensie
    try:
        with open(cand, "rb") as f:
            data = f.read()
    except OSError:
        raise HTTPException(status_code=404)
    return Response(content=data, media_type=mime or "text/html",
                    headers={"Cache-Control": cache_hdr})

app.mount('/', StaticFiles(directory='static', html=True), name='site')

