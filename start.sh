#!/bin/sh
# Rep Agent: uvicorn (bot + dashboard) + optioneel de WhatsApp-bridge in één container.
set -e

export BACKEND_URL="http://127.0.0.1:${PORT:-7860}"
export BACKEND_KEY="${ACCESS_CODE}"
export WA_SESSION_DIR="${WA_SESSION_DIR:-/app/data/whatsapp-session}"

if [ "${WHATSAPP_ENABLED:-0}" = "1" ]; then
  echo "[start] WhatsApp-bridge starten (sessie: $WA_SESSION_DIR)"
  node /app/whatsapp-bridge/bridge.js &
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}"
