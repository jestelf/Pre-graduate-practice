@echo off
REM ───────── Запускаем FastAPI (API + StaticFiles) ─────────
start "VoiceNet API" cmd /k ^
  uvicorn api:app --host 0.0.0.0 --port 7860

REM ───────── LocalTunnel (Node.js + npm) ─────────
start "VoiceNet Tunnel" cmd /k ^
  npx localtunnel --port 7860 --subdomain voicenet
