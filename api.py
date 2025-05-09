"""
VoiceNet Classifier API + Demo (v1.3)
────────────────────────────────────────────────────────────────────────────
• GET  /            → static/index.html
• GET  /static/...   → ваши ассеты
• POST /predict?key=… → загрузка аудиофайла, возврат JSON
"""
import os, time, tempfile, pathlib, contextlib, logging, traceback, warnings
import numpy as np
import librosa, torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ─── 1. Константы ──────────────────────────────────────────────────────────
API_KEY   = "voicenet2024"
THRESHOLD = 0.18
MAX_BYTES = 5 * 1024 * 1024
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

# ─── 2. Гиперпараметры ─────────────────────────────────────────────────────
SAMPLE_RATE = 16000; N_MELS = 80
PATCH_TIME, PATCH_FREQ = 4, 4
D_MODEL = 256; NUM_HEADS = 4; NUM_LAYERS = 4
DIM_FEEDFORWARD = 512; SEGMENT_SIZE = 8; NUM_SEGMENT_LAYERS = 2
NUM_CLASSES = 3

def compute_mel_spectrogram(path, sr=SAMPLE_RATE, n_mels=N_MELS):
    y, sr = librosa.load(path, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db, dtype=torch.float).T

def compute_artifact_map(mel_spec, k=9):
    x = mel_spec.unsqueeze(0).transpose(1, 2)
    return (x - F.avg_pool1d(x, k, stride=1, padding=k//2))\
             .transpose(1, 2).squeeze(0)

# ─── 3. Архитектура ────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, d_model, pt, pf):
        super().__init__()
        self.pt, self.pf, self.dim = pt, pf, pt * pf
        self.proj = nn.Linear(self.dim, d_model)
    def forward(self, x):
        B, T, F = x.shape
        T = (T // self.pt) * self.pt; F = (F // self.pf) * self.pf
        x = x[:, :T, :F].view(B, T//self.pt, self.pt, F//self.pf, self.pf)
        x = x.permute(0,1,3,2,4).contiguous().view(B, -1, self.dim)
        return self.proj(x)

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, d, h, L, f):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, h, f, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, L)
        self.cls_token  = nn.Parameter(torch.zeros(1,1,d))
        self.pos_embed  = nn.Parameter(torch.zeros(1,10000,d))
    def forward(self, xm, xa):
        B = xm.size(0)
        x = torch.cat([self.cls_token.repeat(B,1,1), xm, xa], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        out = self.transformer(x)
        return out[:,0], out[:,1:]

class SegmentLevelSelfAttention(nn.Module):
    def __init__(self, d, heads=2, layers=2, f=256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, f, batch_first=True)
        self.seg_enc = nn.TransformerEncoder(layer, layers)
    def forward(self, tok):
        B,N,D = tok.shape; n = N // SEGMENT_SIZE
        seg = tok[:, :n*SEGMENT_SIZE, :].view(B, n, SEGMENT_SIZE, D).mean(2)
        return self.seg_enc(seg).mean(1)

class MultiTaskHeads(nn.Module):
    def __init__(self, d, nc):
        super().__init__()
        h = d // 2
        self.bin_fc = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,1))
        self.mul_fc = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,nc))
    def forward(self, x):
        return self.bin_fc(x).squeeze(-1), self.mul_fc(x)

class PatentTTSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_emb = PatchEmbed(D_MODEL, PATCH_TIME, PATCH_FREQ)
        self.art_emb = PatchEmbed(D_MODEL, PATCH_TIME, PATCH_FREQ)
        self.enc     = TransformerEncoderWrapper(D_MODEL, NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD)
        self.seg     = SegmentLevelSelfAttention(D_MODEL, 2, NUM_SEGMENT_LAYERS)
        self.heads   = MultiTaskHeads(2*D_MODEL, NUM_CLASSES)
    def forward(self, mel, art):
        cls, tok = self.enc(self.mel_emb(mel), self.art_emb(art))
        return self.heads(torch.cat([cls, self.seg(tok)], dim=-1))

# ─── 4. Загрузка весов ─────────────────────────────────────────────────────
model = PatentTTSNet().to(DEVICE)
model.load_state_dict(torch.load("models/patent_tts_net.pth", map_location=DEVICE))
model.eval()

# ─── 5. FastAPI ────────────────────────────────────────────────────────────
app = FastAPI(title="VoiceNet Classifier", version="1.2")
logger = logging.getLogger("uvicorn.error")

# 1) отдаём статические файлы из ./static по /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2) корневой путь возвращает index.html
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

def auth(q, h): return q == API_KEY or h == API_KEY

@app.exception_handler(Exception)
async def global_err(req, exc):
    logger.error("Unhandled error: %s\n%s", exc, traceback.format_exc())
    raise HTTPException(500, "Internal server error")

@app.post("/predict", response_class=JSONResponse)
async def predict(
    file: UploadFile = File(...),
    key : str | None = Query(None),
    x_api_key: str | None = Header(None, alias="X-API-Key")
):
    # Авторизация
    if not auth(key, x_api_key):
        raise HTTPException(401, "Invalid or missing API key")
    # MIME-проверка
    ct = file.content_type or ""
    if not (ct.startswith("audio/") or ct == "application/octet-stream"):
        raise HTTPException(400, f"Unsupported content-type {ct}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (>5 MB)")

    # Временный файл (Windows-safe)
    ext = pathlib.Path(file.filename).suffix or ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(data); tmp.close()
        mel = compute_mel_spectrogram(tmp.name).to(DEVICE)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp.name)
    art = compute_artifact_map(mel).to(DEVICE)

    # Инференс
    t0 = time.perf_counter()
    with torch.no_grad():
        logb, logm = model(mel.unsqueeze(0), art.unsqueeze(0))
    dt = (time.perf_counter() - t0)*1000

    prob = torch.sigmoid(logb)[0].item()
    names = ["original","synth_same_text","synth_random_text"]
    return {
        "binary_prediction"    : "Fake" if prob > THRESHOLD else "Real",
        "binary_score"         : round(prob,3),
        "binary_threshold"     : THRESHOLD,
        "multiclass_prediction": names[int(torch.argmax(logm))],
        "multiclass_logits"    : [round(v,3) for v in logm[0].cpu().tolist()],
        "inference_time_ms"    : round(dt,1)
    }
