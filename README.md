---
title: Rep Finance Bot
emoji: 💶
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Rep Finance — Telegram voice bot + dashboard

Stuur Nederlandse stemberichten naar een Telegram-bot; hij transcribeert ze,
zet ze om naar een boekhoud-actie via je GLM API, en werkt een kasboek bij dat
op een live website staat: **cash vs basetao-portemonnee vs winstmarge**.

## Wat het doet

1. **Telegram voice (NL)** → `faster-whisper` (lokaal op de Space, geen API-kosten)
2. Transcript → **GLM** (OpenAI-compatibel) → strikte JSON:
   `{type: income|cost|query|note, amount_eur, method: cash|basetao, customer, items, note}`
3. Actie → `data/ledger.json` (+ optionele mirror naar een privé HF Dataset)
4. Bot antwoordt in het Nederlands met de bijgewerkte splitsing; de website op `/`
   laat bars zien voor cash%, basetao%, winstmarge en de **gap = cash% − marge%**
   (jouw doel: gap ≈ 0, dus het contante-inkomsten-aandeel volgt je winstaandeel).

## Deployen op Hugging Face (gratis)

1. **BotFather**: maak op Telegram een bot met `/newbot`, kopieer de token.
2. **HF account** (gratis) → [huggingface.co/new-space](https://huggingface.co/new-space)
   → naam bijv. `rep-finance` → SDK: **Docker** → Hardware: **CPU basic (free)**.
3. Upload deze map (of `git push` naar de Space-repo): `app.py`, `Dockerfile`,
   `requirements.txt`, `data/ledger.json`, `keepalive.py`, deze `README.md`.
4. Space → **Settings → Variables and secrets**, voeg toe:
   | Secret | Waarde |
   |---|---|
   | `TELEGRAM_TOKEN` | van BotFather |
   | `LLM_API_KEY` | je Z.ai key (coding plan) |
   | `LLM_BASE_URL` | `https://api.z.ai/api/coding/paas/v4` (check je plan-docs; gewone API keys: `https://api.z.ai/api/paas/v4`) |
   | `LLM_MODEL` | bijv. `glm-4.6` |
   | `WHISPER_MODEL` | `base` (snel genoeg; `tiny` nog sneller) |
   | `HF_DATASET` (optioneel) | `jouwnaam/rep-ledger` (maak privé-dataset aan) |
   | `HF_TOKEN` (optioneel) | HF write-token, alleen nodig voor de dataset-backup |
5. Herstart de Space. De website staat op `https://<gebruiker>-<space>.hf.space/`.
6. Stuur een stembericht: *"Koppig heeft 40 euro cash gegeven voor het Ajax setje"*.

## 24/7 houden

Gratis Spaces slapen na **48 uur zonder verkeer**. Oplossingen (kies één):
- **UptimeRobot / cron-job.org**: monitor op `https://<je-space>.hf.space/healthz`, elke 10 min.
- **GitHub Action** (gratis, op elke repo):
  ```yaml
  name: keepalive
  on:
    schedule: [{cron: "*/10 * * * *"}]
  jobs:
    ping:
      runs-on: ubuntu-latest
      steps: [{uses: martinbeentjes/npm-get-domain-action@master}, {run: "curl -s https://<je-space>.hf.space/healthz > /dev/null"}]
  ```
  (simpeler: alleen de `run: curl ...` stap)
- **keepalive.py** op een always-on machine: `SPACE_URL=https://... python keepalive.py`

Gaat de Space toch slapen: Telegram bewaart niet-opgehaalde updates **24 uur**,
zodat stemberichten niet verloren gaan — ze worden verwerkt zodra hij wakker is.

## Belangrijk: opslag

De schijf van een Space wordt **gewist bij elke restart**. Daarom:
- Zet `HF_DATASET` + `HF_TOKEN` → het kasboek wordt na elke mutatie naar je
  privé-dataset geschreven en bij het opstarten teruggelezen. Dat is de backup.
- Alternatief: download `data/ledger.json` handmatig af en toe uit de repo.

## Dashboard

- `https://<je-space>.hf.space/` — bars: cash vs basetao, winstmarge, gap
- `/stats` — JSON (voor eigen scripts)
- `/api/entry` — POST `{"type":"income","amount_eur":40,"method":"cash","customer":"Koppig"}`
- Op de site staat ook een invoerformulier voor handmatige mutaties.
