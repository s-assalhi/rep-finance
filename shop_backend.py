"""YZ Shop webshop-backend — router wordt door app.py geïmporteerd (regelnr ±1653).

Publiek:  POST /shop/intent (winkelmand vastzetten) · GET /shop/intent/{id} ·
          GET /shop/checkout (checkout-formulier static/checkout_winkel.html) ·
          POST /shop/order (order + token) · GET /shop/pay/{num} (iDEAL-start) ·
          GET /shop/order/{num} (statuspagina: HTML-shell of JSON, token-check).
Beheer:   GET /api/shop/data · POST /api/shop/order/{num} (acties) ·
          GET /api/shop/export?type=orders|klanten (CSV).
          Deze /api/*-routes zitten óók al achter de access_gate van app.py;
          hier dubbel checken mag (zie _admin_ok).

BELANGRIJK: app.py importeert dit module middenin z'n eigen uitvoer — dus alle
helpers uit app.py LAZY ophalen (pas ín de functies via _app()).
"""
import csv
import hashlib
import io
import math
import os
import re
import secrets
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response)

router = APIRouter()

INTENT_TTL = 24 * 3600.0  # intent is 24 uur geldig


def _app():
    import app  # lazy: pas op runtime, anders circular import tijdens het laden van app.py
    return app


def _jerr(msg, status=400):
    return JSONResponse({"ok": False, "error": msg}, status_code=status)


def _winkel_page(bestand):
    """Statische winkelpagina serven; bestaat hij nog niet, dan nette placeholder i.p.v. 500."""
    pad = os.path.join("static", bestand)
    if os.path.isfile(pad):
        return FileResponse(pad)
    return HTMLResponse(
        '<!doctype html><html lang="nl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>YZ Shop</title></head><body style="font-family:system-ui,sans-serif;'
        'background:#0c0d10;color:#f2f3f5;display:flex;align-items:center;'
        'justify-content:center;min-height:100vh;margin:0">'
        "<p>Pagina laadt&hellip; verversen?</p></body></html>")


def _intent_vervallen(it):
    ts = str((it or {}).get("ts") or "")
    try:
        t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return True
    return time.time() - t > INTENT_TTL


def _admin_ok(request: Request):
    """Dubbele key-check naast de access_gate van app.py (zelfde semantiek:
    geen ACCESS_CODE ingesteld = open, bijv. lokaal)."""
    ap = _app()
    code = ap.ACCESS_CODE
    if not code:
        return True
    got = (request.query_params.get("key")
           or request.headers.get("x-access-code")
           or request.cookies.get("key"))
    return got == code


def _order_bedrag(o):
    """Betaalbedrag in hele euro's (prijzen zijn al heel; ceil vangt float-ruis af)."""
    try:
        return int(math.ceil(float(o.get("price_eur") or 0)))
    except (TypeError, ValueError):
        return 0


def _adres_tekst(a):
    if not a:
        return ""
    s = (str(a.get("straat") or "") + " " + str(a.get("huisnr") or "") + ", "
         + str(a.get("postcode") or "") + " " + str(a.get("plaats") or "")).strip()
    land = a.get("land") or ""
    if land and land != "NL":
        s += " (" + str(land) + ")"
    return s.strip(", ")


def _cny_float(s):
    """Basetao-saldotekst ('1,941.47' of '1.941,47') naar float; None bij twijfel."""
    s = str(s or "").strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")  # 1.941,47 (NL-stijl)
        else:
            s = s.replace(",", "")  # 1,941.47 (US-stijl)
    elif "," in s:
        s = s.replace(",", ".") if re.fullmatch(r"\d+,\d{1,2}", s) else s.replace(",", "")
    elif "." in s:
        s = s if re.fullmatch(r"\d+\.\d{1,2}", s) else s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def _wa_ping(ap, msg):
    try:
        ap._stuur_via_bridge(ap.WA_SELF_JID, msg)
    except Exception:  # noqa: BLE001
        pass


# ---------------- publiek: winkelmand-intent ----------------

@router.post("/shop/intent")
async def shop_intent(request: Request):
    ap = _app()
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _jerr("Ongeldige aanvraag (geen JSON).")
    items = body.get("items")
    if not isinstance(items, list) or not items:
        return _jerr("Je winkelmand is leeg.")
    clean, totaal = [], 0
    for it in items:
        if not isinstance(it, dict):
            return _jerr("Ongeldig item in je winkelmand.")
        try:
            prijs = int(round(float(it.get("prijs") or 0)))
            qty = int(round(float(it.get("qty") or 0)))
        except (TypeError, ValueError):
            return _jerr("Ongeldige prijs of aantal in je winkelmand.")
        titel = str(it.get("titel") or "").strip()
        if not titel or prijs <= 0 or qty <= 0:
            return _jerr("Ongeldig item in je winkelmand.")
        clean.append({"id": str(it.get("id") or "").strip(), "titel": titel,
                      "prijs": prijs, "qty": qty})
        totaal += prijs * qty
    if totaal < 10:
        return _jerr("De minimale bestelling is €10.")
    if totaal > 5000:
        return _jerr("De maximale bestelling is €5000.")
    intent_id = "IN" + secrets.token_hex(8)
    with ap._ledlock:
        d = ap.ledger_load()
        d.setdefault("shop", {}).setdefault("intents", {})[intent_id] = {
            "ts": ap._now(), "items": clean, "totaal": totaal}
        ap.ledger_save(d)
    return JSONResponse({"ok": True, "intent_id": intent_id, "totaal": totaal})


@router.get("/shop/intent/{intent_id}")
def shop_intent_get(intent_id: str):
    ap = _app()
    with ap._ledlock:
        it = ap.ledger_load().get("shop", {}).get("intents", {}).get(intent_id)
    if not it or _intent_vervallen(it):
        return _jerr("Bestelling verlopen of onbekend — maak hem opnieuw aan.", 404)
    return JSONResponse({"ok": True, "items": it.get("items") or [],
                         "totaal": int(it.get("totaal") or 0)})


# ---------------- publiek: checkout-formulier + order + betaling ----------------

@router.get("/shop-dashboard", include_in_schema=False)
def shop_dashboard():
    # Shopify-achtig beheer; achter de access_gate van app.py (pad staat in de gate-lijst)
    return _winkel_page("shop_dashboard.html")


@router.get("/shop/checkout", include_in_schema=False)
def shop_checkout():
    # formulier-pagina van de frontend; bestaat nog niet -> placeholder
    return _winkel_page("checkout_winkel.html")


@router.post("/shop/order")
async def shop_order(request: Request):
    ap = _app()
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _jerr("Ongeldige aanvraag (geen JSON).")

    def veld(k):
        return str(body.get(k) or "").strip()

    intent_id = veld("intent_id")
    naam, email = veld("naam"), veld("email").lower()
    telefoon, straat = veld("telefoon"), veld("straat")
    huisnr, postcode, plaats = veld("huisnr"), veld("postcode"), veld("plaats")
    land = veld("land") or "NL"
    notitie, marketing = veld("notitie"), bool(body.get("marketing"))
    if not intent_id:
        return _jerr("Geen bestelling meegegeven — start opnieuw in de winkel.")
    for label, waarde in (("naam", naam), ("straat", straat), ("huisnummer", huisnr),
                          ("postcode", postcode), ("plaats", plaats)):
        if not waarde:
            return _jerr("Vul je " + label + " in.")
    if "@" not in email or "." not in email:
        return _jerr("Vul een geldig e-mailadres in.")
    with ap._ledlock:
        d = ap.ledger_load()
        intent = d.get("shop", {}).get("intents", {}).get(intent_id)
        if not intent or _intent_vervallen(intent):
            return _jerr("Bestelling verlopen of onbekend — maak hem opnieuw aan.", 404)
        items = intent.get("items") or []
        totaal = int(intent.get("totaal") or 0)
        o = ap.order_add(d, naam, "; ".join(f"{it.get('titel')} ×{it.get('qty')}" for it in items),
                         totaal, payment_method="ideal")
        num = o["num"]
        token = hashlib.sha256(f"{num}|{email}|{ap.ACCESS_CODE}".encode("utf-8")).hexdigest()[:16]
        o["source"] = "webshop"
        o["webshop"] = {
            "email": email, "telefoon": telefoon,
            "adres": {"straat": straat, "huisnr": huisnr, "postcode": postcode,
                      "plaats": plaats, "land": land},
            "items": items, "marketing": marketing, "notitie": notitie,
            "token": token, "betaal_status": "wacht"}
        c = d.setdefault("shop", {}).setdefault("customers", {}).setdefault(email, {})
        c.update({"naam": naam, "telefoon": telefoon, "adres": o["webshop"]["adres"],
                  "marketing": marketing, "laatste": ap._now()})
        nums = c.setdefault("orders", [])
        if num not in nums:
            nums.append(num)
        d.get("shop", {}).get("intents", {}).pop(intent_id, None)  # intent is 1x bruikbaar
        ap.ledger_save(d)
        ap.hf_sync_up()
    return JSONResponse({"ok": True, "num": num, "token": token,
                         "status_url": f"/shop/order/{num}?t={token}"})


@router.get("/shop/pay/{num:int}", include_in_schema=False)
def shop_pay(num: int):
    ap = _app()
    with ap._ledlock:
        d = ap.ledger_load()
        o = next((x for x in d.get("orders", [])
                  if x.get("num") == num and x.get("source") == "webshop"), None)
    if not o:
        return ap._checkout_error(404, "Deze bestelling bestaat niet (meer).")
    if (o.get("webshop") or {}).get("betaal_status") == "betaald":
        return ap._checkout_error(400, "Deze bestelling is al betaald.")
    try:
        pay_url = ap.basetao_create_recharge(_order_bedrag(o))
    except Exception as e:  # noqa: BLE001
        print("shop pay error:", e)
        return ap._checkout_error(502, "De betaalpartner accepteert de betaling nu even niet.")
    with ap._ledlock:
        d2 = ap.ledger_load()  # verse stand: intussen kan er iets gewijzigd zijn
        o2 = next((x for x in d2.get("orders", [])
                   if x.get("num") == num and x.get("source") == "webshop"), None)
        if o2 is not None and (o2.get("webshop") or {}).get("betaal_status") != "betaald":
            w2 = o2.setdefault("webshop", {})
            w2["betaal_status"] = "gestart"
            w2["betaal_gestart"] = ap._now()
            ap.ledger_save(d2)
    # GEEN topup-entry boeken hier: dit geld is de betaling van een echte shop-order,
    # inkomsten worden pas geboekt bij bevestiging ("betaald" via /api/shop/order/{num}).
    # Anders dubbel met de bestaande /checkout-topup-logica van app.py.
    threading.Thread(target=ap.hf_sync_up, daemon=True).start()
    return RedirectResponse(pay_url, status_code=302)


@router.get("/shop/order/{num:int}", include_in_schema=False)
def shop_order_status(num: int, request: Request, format: str = ""):
    # content-negotiation: browser-navigatie -> HTML-shell; fetch (?format=json of
    # Accept: application/json of generieke */*) -> JSON mét token-check
    accept = request.headers.get("accept") or ""
    if not (format == "json" or "application/json" in accept or "text/html" not in accept):
        return _winkel_page("order_status_winkel.html")
    ap = _app()
    with ap._ledlock:
        d = ap.ledger_load()
        o = next((x for x in d.get("orders", [])
                  if x.get("num") == num and x.get("source") == "webshop"), None)
    if not o:
        return _jerr("Bestelling niet gevonden.", 404)
    w = o.get("webshop") or {}
    tok = request.query_params.get("t") or request.headers.get("t") or ""
    if tok != w.get("token"):
        return _jerr("Ongeldige link — gebruik de link uit je bevestigingsmail/chat.", 403)
    return JSONResponse({"ok": True, "order": {
        "num": num, "naam": o.get("customer"), "items": w.get("items") or [],
        "totaal": _order_bedrag(o), "betaal_status": w.get("betaal_status"),
        "order_status": o.get("order_status"), "updated": o.get("updated")}})


# ---------------- beheer: dashboard-data, acties, export ----------------

def _webshop_orders(d):
    uit = [o for o in d.get("orders", []) if o.get("source") == "webshop"]
    # nieuwste eerst; bij dezelfde seconde wint het hoogste ordernummer
    uit.sort(key=lambda o: (o.get("created") or "", o.get("num") or 0), reverse=True)
    return uit


def _verkorte_order(o):
    w = o.get("webshop") or {}
    return {"num": o.get("num"), "created": o.get("created"), "updated": o.get("updated"),
            "naam": o.get("customer"), "items": w.get("items") or [],
            "totaal": _order_bedrag(o), "betaal_status": w.get("betaal_status"),
            "order_status": o.get("order_status")}


@router.get("/api/shop/data")
def shop_data(request: Request):
    ap = _app()
    if not _admin_ok(request):
        return _jerr("toegang geweigerd: voeg ?key=... toe", 401)
    with ap._ledlock:
        d = ap.ledger_load()
    ws_orders = _webshop_orders(d)
    orders_uit = []
    for o in ws_orders:
        w = o.get("webshop") or {}
        orders_uit.append({
            "num": o.get("num"), "created": o.get("created"), "updated": o.get("updated"),
            "naam": o.get("customer"), "email": w.get("email"), "telefoon": w.get("telefoon"),
            "adres": w.get("adres"), "items": w.get("items") or [],
            "totaal": _order_bedrag(o), "betaal_status": w.get("betaal_status"),
            "order_status": o.get("order_status"), "marketing": bool(w.get("marketing")),
            "notitie": w.get("notitie") or "",
            "vermoed_betaald": w.get("vermoed_betaald") or None})
    klanten_uit = []
    for email, c in sorted((d.get("shop", {}).get("customers") or {}).items()):
        besteed = sum(_order_bedrag(o) for o in ws_orders
                      if (o.get("webshop") or {}).get("email") == email)
        klanten_uit.append({
            "email": email, "naam": c.get("naam"), "telefoon": c.get("telefoon"),
            "adres": c.get("adres"), "aantal_orders": len(c.get("orders") or []),
            "totaal_besteed": besteed, "marketing": bool(c.get("marketing")),
            "laatste": c.get("laatste")})
    # analytics
    totaal_alle = sum(x["totaal"] for x in orders_uit)
    omzet_betaald = sum(x["totaal"] for x in orders_uit if x["betaal_status"] == "betaald")
    n = len(orders_uit)
    n_betaald = sum(1 for x in orders_uit if x["betaal_status"] == "betaald")
    dagmap = {}
    for o in ws_orders:
        dag = str(o.get("created") or "")[:10]
        if not dag:
            continue
        r = dagmap.setdefault(dag, {"omzet_betaald": 0, "orders": 0})
        r["orders"] += 1
        if (o.get("webshop") or {}).get("betaal_status") == "betaald":
            r["omzet_betaald"] += _order_bedrag(o)
    per_dag = []
    for i in range(13, -1, -1):
        dag = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
        r = dagmap.get(dag) or {}
        per_dag.append({"dag": dag, "omzet_betaald": r.get("omzet_betaald", 0),
                        "orders": r.get("orders", 0)})
    prod = {}
    for o in ws_orders:
        for it in ((o.get("webshop") or {}).get("items") or []):
            try:
                q = int(round(float(it.get("qty") or 1)))
                p = int(round(float(it.get("prijs") or 0)))
            except (TypeError, ValueError):
                continue
            r = prod.setdefault(str(it.get("titel") or "?"), {"qty": 0, "omzet": 0})
            r["qty"] += q
            r["omzet"] += q * p
    top = sorted(prod.items(), key=lambda kv: kv[1]["qty"], reverse=True)[:8]
    return JSONResponse({"ok": True, "orders": orders_uit, "customers": klanten_uit,
                         "analytics": {
                             "omzet_betaald": omzet_betaald,
                             "omzet_open": totaal_alle - omzet_betaald,
                             "aantal_orders": n,
                             "gem_orderwaarde": round(totaal_alle / n, 2) if n else 0,
                             "betaald_pct": round(100.0 * n_betaald / n, 1) if n else 0.0,
                             "per_dag": per_dag,
                             "top_producten": [{"titel": t, "qty": r["qty"], "omzet": r["omzet"]}
                                               for t, r in top],
                             "aantal_klanten": len(klanten_uit),
                             "marketing_emails": sum(1 for k in klanten_uit if k["marketing"]),
                         }})


_ACTIE_STATUS = {"besteld": "besteld", "onderweg": "onderweg", "binnen": "binnen",
                 "klaar": "klaar", "annuleren": "geen_interesse"}


@router.post("/api/shop/order/{num:int}")
async def shop_order_actie(num: int, request: Request):
    ap = _app()
    if not _admin_ok(request):
        return _jerr("toegang geweigerd: voeg ?key=... toe", 401)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _jerr("Ongeldige aanvraag (geen JSON).")
    actie = str(body.get("action") or "").strip()
    msg = None
    with ap._ledlock:
        d = ap.ledger_load()
        o = next((x for x in d.get("orders", [])
                  if x.get("num") == num and x.get("source") == "webshop"), None)
        if not o:
            return _jerr("Webshop-order niet gevonden.", 404)
        w = o.setdefault("webshop", {})
        if actie == "betaald":
            totaal = _order_bedrag(o)
            ap.order_update(d, num, payment_status="betaald", payment_method="basetao",
                            order_status="te_bestellen")
            w["betaal_status"] = "betaald"
            # income pas bij bevestiging boeken (niet bij het maken van de betaal-link):
            # anders dubbel met de bestaande /checkout-topup-logica van app.py
            d.setdefault("entries", []).append({
                "ts": ap._now(), "type": "income", "amount_eur": totaal,
                "method": "basetao", "customer": o.get("customer"), "items": o.get("items"),
                "note": f"webshop order #{num} (iDEAL)", "source": "webshop"})
            ap.ledger_save(d)
            ap.hf_sync_up()
            msg = f"🛒 Webshop order #{num} betaald: {o.get('customer')} — €{totaal:g}"
        elif actie in _ACTIE_STATUS:
            ap.order_update(d, num, order_status=_ACTIE_STATUS[actie])
            ap.ledger_save(d)
            ap.hf_sync_up()
        else:
            return _jerr("Onbekende actie — gebruik betaald, besteld, onderweg, binnen, "
                         "klaar of annuleren.")
        out = _verkorte_order(o)
    if msg:  # best effort WhatsApp-ping naar jezelf; blokkeert/breekt nooit de response
        threading.Thread(target=_wa_ping, args=(ap, msg), daemon=True).start()
    return JSONResponse({"ok": True, "order": out})


@router.get("/api/shop/export")
def shop_export(request: Request, type: str = "orders"):
    ap = _app()
    if not _admin_ok(request):
        return _jerr("toegang geweigerd: voeg ?key=... toe", 401)
    with ap._ledlock:
        d = ap.ledger_load()
    ws_orders = sorted([o for o in d.get("orders", []) if o.get("source") == "webshop"],
                       key=lambda o: (o.get("created") or "", o.get("num") or 0))
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    if type == "klanten":
        w.writerow(["email", "naam", "telefoon", "adres", "aantal_orders",
                    "totaal_besteed", "marketing"])
        for email, c in sorted((d.get("shop", {}).get("customers") or {}).items()):
            besteed = sum(_order_bedrag(o) for o in ws_orders
                          if (o.get("webshop") or {}).get("email") == email)
            w.writerow([email, c.get("naam") or "", c.get("telefoon") or "",
                        _adres_tekst(c.get("adres") or {}), len(c.get("orders") or []),
                        besteed, "ja" if c.get("marketing") else "nee"])
        bestand = "shop_klanten.csv"
    else:
        w.writerow(["num", "datum", "naam", "email", "telefoon", "adres", "items",
                    "totaal", "betaal_status", "order_status", "marketing"])
        for o in ws_orders:
            wo = o.get("webshop") or {}
            w.writerow([o.get("num"), o.get("created"), o.get("customer") or "",
                        wo.get("email") or "", wo.get("telefoon") or "",
                        _adres_tekst(wo.get("adres") or {}), o.get("items") or "",
                        _order_bedrag(o), wo.get("betaal_status") or "",
                        o.get("order_status") or "",
                        "ja" if wo.get("marketing") else "nee"])
        bestand = "shop_orders.csv"
    return Response(content="\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{bestand}"'})


# ---------------- achtergrond: wallet-controle (vermoed-betaald-detectie) ----------------

def _gestart_zonder_vermoeden(d):
    uit = []
    for o in d.get("orders", []):
        w = o.get("webshop") or {}
        if (o.get("source") == "webshop" and w.get("betaal_status") == "gestart"
                and not w.get("vermoed_betaald")):
            uit.append(o)
    return uit


def _wallet_check():
    ap = _app()
    with ap._ledlock:
        d = ap.ledger_load()
        if not _gestart_zonder_vermoeden(d):
            return
    # HTTP buiten de ledger-lock: de rest van de bot mag niet blokkeren
    s = ap._bt_session()
    r = s.get(ap.BASETAO_BASE + "/my_account/welcome.html", timeout=30,
              params={"r": str(int(time.time() * 1000))})
    m = re.search(r"bi-currency-yen[\s\S]{0,150}?>\s*([\d.,]+)\s*<", r.text)
    saldo = _cny_float(m.group(1)) if m else None
    if saldo is None:
        return  # saldo zat niet in de HTML -> deze cyclus overslaan
    gemeld = None
    with ap._ledlock:
        d = ap.ledger_load()  # verse stand: intussen kan er veel gewijzigd zijn
        gestart = _gestart_zonder_vermoeden(d)
        shop = d.setdefault("shop", {})
        vorige = shop.get("wallet_cny")
        shop["wallet_cny"] = saldo
        shop["wallet_ts"] = ap._now()
        if vorige is not None and gestart:
            stijging = saldo - float(vorige)
            kleinste = min(_order_bedrag(o) for o in gestart)
            if kleinste > 0 and stijging >= kleinste:
                oudste = min(gestart, key=lambda o: o.get("created") or "")
                oudste.setdefault("webshop", {})["vermoed_betaald"] = ap._now()
                gemeld = (oudste["num"], oudste.get("customer") or "?", _order_bedrag(oudste))
        ap.ledger_save(d)
    if gemeld:  # alleen syncen/melden als er iets betekenisvols veranderde
        num, naam, bedrag = gemeld
        threading.Thread(target=_wa_ping, args=(
            ap, f"🛒 Vermoed betaald (check wallet): order #{num} {naam} — €{bedrag:g}"
                " — bevestig in /shop-dashboard"), daemon=True).start()
        threading.Thread(target=ap.hf_sync_up, daemon=True).start()


def shop_startup():
    """Wallet-controle-thread; best effort — een fout slaat 1 cyclus over, crasht nooit."""
    threading.Thread(target=_wallet_loop, daemon=True).start()


def _wallet_loop():
    while True:
        time.sleep(300)
        try:
            _wallet_check()
        except Exception as e:  # noqa: BLE001
            print("shop wallet-check fout:", e)
