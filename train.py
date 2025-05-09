import os
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import librosa
from tqdm import tqdm

# ----------------------------- Гиперпараметры -----------------------------
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
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 3    # 0=original,1=synth_same_text,2=synth_random_text
LAMBDA_ATTR = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Логирование
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========================= 1. Предобработка =========================
def compute_mel_spectrogram(path, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    y, sr = librosa.load(path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(db.T, dtype=torch.float)  # [time, mel_bins]

def compute_artifact_map(mel_spec, kernel_size=9):
    x = mel_spec.unsqueeze(0).transpose(1,2)  # [1, mel, time]
    pad = kernel_size // 2
    smooth = F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)
    art = x - smooth
    return art.transpose(1,2).squeeze(0)       # [time, mel]

# ========================= 2. Датасет =========================
class MultiTaskDataset(Dataset):
    def __init__(self, paths, class_ids):
        self.paths = paths
        self.multi = class_ids
        self.bin = [0 if c==0 else 1 for c in class_ids]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        mel = compute_mel_spectrogram(path)
        art = compute_artifact_map(mel)
        return (
            mel,
            art,
            torch.tensor(self.bin[i], dtype=torch.float),
            torch.tensor(self.multi[i], dtype=torch.long)
        )

def pad_collate_fn(batch):
    mels, arts, bins, multis = zip(*batch)
    max_t = max(m.shape[0] for m in mels)
    def pad(x):
        return F.pad(x, (0,0,0,max_t-x.shape[0]))
    mels = torch.stack([pad(m) for m in mels])
    arts = torch.stack([pad(a) for a in arts])
    bins = torch.stack(bins)
    multis = torch.stack(multis)
    return mels, arts, bins, multis

# ========================= 3. Архитектура =========================
class PatchEmbed(nn.Module):
    def __init__(self, d_model, pt, pf):
        super().__init__()
        self.pt, self.pf = pt, pf
        self.dim = pt * pf
        self.proj = nn.Linear(self.dim, d_model)
    def forward(self, x):
        B,T,F = x.shape
        T0 = (T//self.pt)*self.pt
        F0 = (F//self.pf)*self.pf
        x = x[:,:T0,:F0]
        nT, nF = T0//self.pt, F0//self.pf
        x = x.view(B,nT,self.pt,nF,self.pf)
        x = x.permute(0,1,3,2,4).contiguous().view(B,nT*nF,self.dim)
        return self.proj(x)

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dim_ff):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, nlayers)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1,10000,d_model))
    def forward(self, xm, xa):
        B = xm.size(0)
        cls = self.cls_token.repeat(B,1,1)
        x = torch.cat([cls,xm,xa], dim=1)
        L = x.size(1)
        x = x + self.pos_embed[:,:L,:]
        out = self.transformer(x)
        return out[:,0], out[:,1:]

class SegmentLevelSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dim_ff):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.segment_encoder = nn.TransformerEncoder(layer, nlayers)
    def forward(self, tokens):
        B,N,D = tokens.shape
        c = N // SEGMENT_SIZE
        t = tokens[:,:c*SEGMENT_SIZE,:].view(B,c,SEGMENT_SIZE,D)
        e = t.mean(2)                    # [B, c, D]
        out = self.segment_encoder(e)    # [B, c, D]
        return out.mean(1)               # [B, D]

class MultiTaskHeads(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.bin_fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
        self.multi_fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, num_classes)
        )
    def forward(self, x):
        return self.bin_fc(x).squeeze(-1), self.multi_fc(x)

class PatentTTSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_embed         = PatchEmbed(D_MODEL, PATCH_TIME, PATCH_FREQ)
        self.art_embed         = PatchEmbed(D_MODEL, PATCH_TIME, PATCH_FREQ)
        self.transformer_enc   = TransformerEncoderWrapper(D_MODEL, NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD)
        self.segment_analyzer  = SegmentLevelSelfAttention(D_MODEL, NUM_HEADS, NUM_SEGMENT_LAYERS, DIM_FEEDFORWARD//2)
        self.heads             = MultiTaskHeads(2*D_MODEL, NUM_CLASSES)
    def forward(self, mel, art):
        xm       = self.mel_embed(mel)
        xa       = self.art_embed(art)
        cls, toks= self.transformer_enc(xm, xa)
        seg      = self.segment_analyzer(toks)
        return self.heads(torch.cat([cls, seg], dim=1))

# ========================= 4. Лосс и обучение =========================
def multi_task_loss(logit_bin, logit_multi, lb_bin, lb_multi):
    loss_bin   = F.binary_cross_entropy_with_logits(logit_bin, lb_bin)
    loss_multi = F.cross_entropy(logit_multi, lb_multi)
    return loss_bin + LAMBDA_ATTR * loss_multi

def load_dataset():
    safe   = glob("data/safe/**/*.wav", recursive=True)
    unsafe = glob("data/unsafe/**/*.wav", recursive=True)
    paths  = safe + unsafe
    labels = [0]*len(safe) + [1]*len(unsafe)
    return train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)

def train():
    tr_p, vl_p, tr_lbl, vl_lbl = load_dataset()
    tr_ds = MultiTaskDataset(tr_p, tr_lbl)
    vl_ds = MultiTaskDataset(vl_p, vl_lbl)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    device= DEVICE
    model = PatentTTSNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        for mel, art, lb_bin, lb_multi in tqdm(tr_ld, desc=f"Epoch {epoch}"):
            mel, art, lb_bin, lb_multi = mel.to(device), art.to(device), lb_bin.to(device), lb_multi.to(device)
            logit_bin, logit_multi = model(mel, art)
            loss = multi_task_loss(logit_bin, logit_multi, lb_bin, lb_multi)
            opt.zero_grad(); loss.backward(); opt.step()
        logger.info(f"Epoch {epoch} complete")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/patent_tts_net.pth")
    logger.info("Model saved to models/patent_tts_net.pth")

if __name__ == "__main__":
    train()
