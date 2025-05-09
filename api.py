import io, time, tempfile, warnings
from typing import Literal, List

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query, Header
from fastapi.responses import JSONResponse

API_KEY = "voicenet2024"
THRESHOLD = 0.18
MAX_BYTES = 5 * 1024 * 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_RATE = 16000
N_MELS = 80
PATCH_TIME = 4
PATCH_FREQ = 4
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
SEGMENT_SIZE = 8
NUM_SEGMENT_LAYERS = 2
NUM_CLASSES = 3

def compute_mel_spectrogram(path_or_buf, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    y, sr = librosa.load(path_or_buf, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db, dtype=torch.float).T

def compute_artifact_map(mel_spec, kernel_size=9):
    x = mel_spec.unsqueeze(0).transpose(1, 2)
    smooth = F.avg_pool1d(x, kernel_size, stride=1, padding=kernel_size // 2)
    return (x - smooth).transpose(1, 2).squeeze(0)

class PatchEmbed(nn.Module):
    def __init__(self, d_model, pt, pf):
        super().__init__(); self.pt, self.pf = pt, pf
        self.dim = pt * pf; self.proj = nn.Linear(self.dim, d_model)
    def forward(self, x):
        B,T,F = x.shape
        T2=(T//self.pt)*self.pt; F2=(F//self.pf)*self.pf
        x=x[:,:T2,:F2].view(B,T2//self.pt,self.pt,F2//self.pf,self.pf)
        x=x.permute(0,1,3,2,4).contiguous().view(B,-1,self.dim)
        return self.proj(x)

class TransformerEncoderWrapper(nn.Module):
    def __init__(self,d,h,l,f):
        super().__init__()
        enc=nn.TransformerEncoderLayer(d,h,f,batch_first=True)
        self.tr=nn.TransformerEncoder(enc,l)
        self.cls=nn.Parameter(torch.zeros(1,1,d))
        self.pos=nn.Parameter(torch.zeros(1,10000,d))
    def forward(self,xm,xa):
        B=xm.size(0); x=torch.cat([self.cls.repeat(B,1,1),xm,xa],1)
        x=x+self.pos[:,:x.size(1),:]; out=self.tr(x)
        return out[:,0], out[:,1:]

class SegmentSA(nn.Module):
    def __init__(self,d,heads=2,layers=2,f=256):
        super().__init__()
        enc=nn.TransformerEncoderLayer(d,heads,f,batch_first=True)
        self.seg=nn.TransformerEncoder(enc,layers)
    def forward(self,tok):
        B,N,D=tok.shape; n=N//SEGMENT_SIZE
        tok=tok[:,:n*SEGMENT_SIZE,:].view(B,n,SEGMENT_SIZE,D).mean(2)
        return self.seg(tok).mean(1)

class Heads(nn.Module):
    def __init__(self,d,num_cls):
        super().__init__()
        half=d//2
        self.bin=nn.Sequential(nn.Linear(d,half),nn.ReLU(),nn.Linear(half,1))
        self.mul=nn.Sequential(nn.Linear(d,half),nn.ReLU(),nn.Linear(half,num_cls))
    def forward(self,x): return self.bin(x).squeeze(-1),self.mul(x)

class PatentTTSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel=PatchEmbed(D_MODEL,PATCH_TIME,PATCH_FREQ)
        self.art=PatchEmbed(D_MODEL,PATCH_TIME,PATCH_FREQ)
        self.enc=TransformerEncoderWrapper(D_MODEL,NUM_HEADS,NUM_LAYERS,DIM_FEEDFORWARD)
        self.seg=SegmentSA(D_MODEL,2,NUM_SEGMENT_LAYERS)
        self.hd=Heads(2*D_MODEL,NUM_CLASSES)
    def forward(self,m,a):
        cls,tok=self.enc(self.mel(m),self.art(a))
        return self.hd(torch.cat([cls,self.seg(tok)],-1))

MODEL_PATH = "models/patent_tts_net.pth"
model = PatentTTSNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

app = FastAPI(title="VoiceNet Classifier", docs_url="/docs", redoc_url=None)

def auth_ok(key_query: str | None, key_header: str | None):
    return key_query == API_KEY or key_header == API_KEY

@app.post("/predict", response_class=JSONResponse)
async def predict(
    file: UploadFile = File(...),
    key: str | None = Query(None),
    x_api_key: str | None = Header(None, alias="X-API-Key")
):
    if not auth_ok(key, x_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    if file.content_type not in {"audio/wav", "audio/x-wav"}:
        raise HTTPException(400, f"Unsupported content-type {file.content_type}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (>5 MB)")
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(data); tmp.flush()
        mel = compute_mel_spectrogram(tmp.name).to(DEVICE)
    art = compute_artifact_map(mel).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logb, logm = model(mel.unsqueeze(0), art.unsqueeze(0))
    dt_ms = (time.perf_counter() - t0) * 1000

    prob = torch.sigmoid(logb)[0].item()
    bin_pred = "Fake" if prob > THRESHOLD else "Real"
    logits = logm[0].cpu().tolist()
    cls_names = ["original", "synth_same_text", "synth_random_text"]
    multi_pred = cls_names[int(torch.argmax(logm))]
    return {
        "binary_prediction": bin_pred,
        "binary_score": round(prob, 3),
        "binary_threshold": THRESHOLD,
        "multiclass_prediction": multi_pred,
        "multiclass_logits": [round(v, 3) for v in logits],
        "inference_time_ms": round(dt_ms, 1)
    }
