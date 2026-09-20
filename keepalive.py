"""Keep-alive pinger: run anywhere with Python that is always on.
Pings your Space every 10 minutes so it never hits the 48h sleep.
Usage:  set SPACE_URL env var, then:  python keepalive.py
"""
import os
import time

import requests

URL = os.environ["SPACE_URL"].rstrip("/")

while True:
    try:
        r = requests.get(URL + "/healthz", timeout=30)
        print(time.strftime("%H:%M:%S"), r.status_code)
    except Exception as e:  # noqa: BLE001
        print("ping failed:", e)
    time.sleep(600)
