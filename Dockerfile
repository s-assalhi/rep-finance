FROM python:3.11-slim

# Node 20 voor de WhatsApp-bridge (Baileys = Node-bibliotheek, met git-dependencies)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bridge-dependencies alvast in een eigen layer (sneller herbouwen)
COPY whatsapp-bridge/package.json whatsapp-bridge/
RUN cd whatsapp-bridge && npm install --omit=dev --no-audit --no-fund

COPY . .
ENV HF_HOME=/tmp/hf
EXPOSE 7860
CMD ["sh", "/app/start.sh"]
