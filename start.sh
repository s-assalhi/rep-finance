#!/bin/sh
# Rep Agent: uvicorn (bot + dashboard) + optioneel de WhatsApp-bridge in één container.
set -e

export BACKEND_URL="http://127.0.0.1:${PORT:-7860}"
export BACKEND_KEY="${ACCESS_CODE}"
export WA_SESSION_DIR="${WA_SESSION_DIR:-/app/data/whatsapp-session}"

if [ "${WHATSAPP_ENABLED:-0}" = "1" ]; then
  (
    # wacht tot uvicorn draait: de startup-event herstelt eerst de WhatsApp-sessie
    # uit de HF-dataset; start de bridge pas daarna (anders boot hij zonder sessie).
    i=0
    while [ $i -lt 60 ]; do
      if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${PORT:-7860}/healthz"; then
        break
      fi
      i=$((i + 1))
      sleep 1
    done
    sleep 3
    echo "[start] WhatsApp-bridge starten (sessie: $WA_SESSION_DIR)"
    node /app/whatsapp-bridge/bridge.js &
  ) &
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}"
