import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

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

def compute_mel_spectrogram(audio_path, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = torch.tensor(mel_db, dtype=torch.float32)
    mel_db = mel_db.transpose(0, 1)
    return mel_db

def compute_artifact_map(mel_spec, kernel_size=9):
    x = mel_spec.unsqueeze(0).transpose(1, 2)
    pad = kernel_size // 2
    smooth = F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)
    art = x - smooth
    art = art.transpose(1, 2).squeeze(0)
    return art

class PatchEmbed(nn.Module):
    def __init__(self, d_model, patch_time, patch_freq):
        super().__init__()
        self.patch_time = patch_time
        self.patch_freq = patch_freq
        self.patch_dim = patch_time * patch_freq
        self.proj = nn.Linear(self.patch_dim, d_model)

    def forward(self, x):
        B, T, F = x.shape
        new_T = (T // self.patch_time) * self.patch_time
        new_F = (F // self.patch_freq) * self.patch_freq
        x = x[:, :new_T, :new_F]
        nT = new_T // self.patch_time
        nF = new_F // self.patch_freq
        x = x.view(B, nT, self.patch_time, nF, self.patch_freq)
        x = x.permute(0,1,3,2,4).contiguous()
        x = x.view(B, nT*nF, self.patch_dim)
        x = self.proj(x)
        return x

class TransformerEncoderWrapper(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 10000, d_model))

    def forward(self, x_mel, x_art):
        B, N_mel, _ = x_mel.shape
        B, N_art, _ = x_art.shape
        cls_tok = self.cls_token.repeat(B,1,1)
        x_all = torch.cat([cls_tok, x_mel, x_art], dim=1)
        seq_len = x_all.size(1)
        x_all = x_all + self.pos_embed[:, :seq_len, :]
        out = self.transformer(x_all)
        cls_emb = out[:, 0, :]
        tokens = out[:, 1:, :]
        return cls_emb, tokens

class SegmentLevelSelfAttention(nn.Module):
    def __init__(self, d_model, nhead=2, num_layers=2, dim_feedforward=256):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.segment_encoder = nn.TransformerEncoder(enc_layer, num_layers)

    def forward(self, enc_tokens):
        B, N, D = enc_tokens.shape
        seg_count = N // SEGMENT_SIZE
        used_len = seg_count * SEGMENT_SIZE
        tokens = enc_tokens[:, :used_len, :]
        tokens = tokens.view(B, seg_count, SEGMENT_SIZE, D)
        seg_emb = tokens.mean(dim=2)
        seg_enc = self.segment_encoder(seg_emb)
        seg_rep = seg_enc.mean(dim=1)
        return seg_rep

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
        logit_bin = self.bin_fc(x).squeeze(-1)
        logit_multi = self.multi_fc(x)
        return logit_bin, logit_multi

class PatentTTSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_embed = PatchEmbed(D_MODEL, PATCH_TIME, PATCH_FREQ)
        self.art_embed = PatchEmbed(D_MODEL, PATCH_TIME, PATCH_FREQ)
        self.transformer_enc = TransformerEncoderWrapper(
            D_MODEL, NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD
        )
        self.segment_analyzer = SegmentLevelSelfAttention(
            D_MODEL, nhead=2, num_layers=NUM_SEGMENT_LAYERS, dim_feedforward=256
        )
        self.heads = MultiTaskHeads(2 * D_MODEL, NUM_CLASSES)

    def forward(self, mel_input, art_input):
        x_mel = self.mel_embed(mel_input)
        x_art = self.art_embed(art_input)
        cls_emb, enc_tokens = self.transformer_enc(x_mel, x_art)
        seg_rep = self.segment_analyzer(enc_tokens)
        fused = torch.cat([cls_emb, seg_rep], dim=-1)
        logit_bin, logit_multi = self.heads(fused)
        return logit_bin, logit_multi

CLASSES = ["original", "synth_same_text", "synth_random_text"]

model = PatentTTSNet()
state = torch.load("models/patent_tts_net.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

def predict(audio_path):
    mel = compute_mel_spectrogram(audio_path)
    art = compute_artifact_map(mel)
    max_len = 400
    T_mel = mel.shape[0]
    if T_mel < max_len:
        pad_T = max_len - T_mel
        mel = F.pad(mel, (0, 0, 0, pad_T))
        art = F.pad(art, (0, 0, 0, pad_T))
    else:
        mel = mel[:max_len, :]
        art = art[:max_len, :]
    mel = mel.unsqueeze(0)
    art = art.unsqueeze(0)
    with torch.no_grad():
        logit_bin, logit_multi = model(mel, art)
        pred_bin = (torch.sigmoid(logit_bin) > 0.5).long().item()
        pred_multi = logit_multi.argmax(dim=1).item()
    bin_str = "real" if pred_bin == 0 else "fake"
    multi_str = CLASSES[pred_multi]
    return f"BINARY: {bin_str}, CLASS: {multi_str}"

def browse_file():
    filepath = filedialog.askopenfilename(
        filetypes=[("Audio files", "*.wav *.ogg *.mp3 *.flac"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    result = predict(filepath)
    result_text.set(f"Результат: {result}")

app = tk.Tk()
app.title("PatentTTSNet: Проверка синтетической речи")
app.geometry("500x250")

frame = tk.Frame(app, padx=20, pady=20)
frame.pack(expand=True)

label = tk.Label(frame, text="Выберите аудиофайл (wav/ogg/mp3/flac...)", font=("Arial", 12))
label.pack(pady=10)

browse_btn = tk.Button(frame, text="📁 Выбрать файл", command=browse_file, font=("Arial", 12))
browse_btn.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(frame, textvariable=result_text, font=("Arial", 14, "bold"), fg="blue")
result_label.pack(pady=20)

app.mainloop()
