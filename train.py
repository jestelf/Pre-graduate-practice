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
EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 3
LAMBDA_ATTR = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def compute_mel_spectrogram(audio_path, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = torch.tensor(mel_db, dtype=torch.float)
    return mel_db.transpose(0, 1)

def compute_artifact_map(mel_spec, kernel_size=9):
    x = mel_spec.unsqueeze(0)
    x = x.transpose(1, 2)
    pad = kernel_size // 2
    smooth = F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)
    art = x - smooth
    art = art.transpose(1, 2).squeeze(0)
    return art

class MultiTaskDataset(Dataset):
    def __init__(self, audio_paths, class_ids):
        super().__init__()
        self.audio_paths = audio_paths
        self.class_ids = class_ids
        self.bin_labels = [0 if c==0 else 1 for c in class_ids]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        multi_label = self.class_ids[idx]
        bin_label = self.bin_labels[idx]
        mel = compute_mel_spectrogram(path)
        art = compute_artifact_map(mel)
        return {
            'mel': mel,
            'art': art,
            'bin_label': bin_label,
            'multi_label': multi_label
        }

def pad_collate_fn(batch):
    mels = [item['mel'] for item in batch]
    arts = [item['art'] for item in batch]
    bin_labels = [item['bin_label'] for item in batch]
    multi_labels = [item['multi_label'] for item in batch]
    max_len = max(m.shape[0] for m in mels)

    def pad_time(x, max_len):
        T, M = x.shape
        pad_T = max_len - T
        if pad_T > 0:
            x = F.pad(x.unsqueeze(0), (0, 0, 0, pad_T)).squeeze(0)
        return x

    mels_padded = [pad_time(m, max_len) for m in mels]
    arts_padded = [pad_time(a, max_len) for a in arts]

    mels_tensor = torch.stack(mels_padded, dim=0)
    arts_tensor = torch.stack(arts_padded, dim=0)
    bin_labels = torch.tensor(bin_labels, dtype=torch.long)
    multi_labels = torch.tensor(multi_labels, dtype=torch.long)

    return mels_tensor, arts_tensor, bin_labels, multi_labels

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
        x = x.permute(0, 1, 3, 2, 4).contiguous()
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
        cls_tok = self.cls_token.repeat(B, 1, 1)
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
        self.transformer_enc = TransformerEncoderWrapper(D_MODEL, NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD)
        self.segment_analyzer = SegmentLevelSelfAttention(D_MODEL, nhead=2, num_layers=NUM_SEGMENT_LAYERS, dim_feedforward=256)
        self.heads = MultiTaskHeads(2 * D_MODEL, NUM_CLASSES)

    def forward(self, mel_input, art_input):
        x_mel = self.mel_embed(mel_input)
        x_art = self.art_embed(art_input)
        cls_emb, enc_tokens = self.transformer_enc(x_mel, x_art)
        seg_rep = self.segment_analyzer(enc_tokens)
        fused = torch.cat([cls_emb, seg_rep], dim=-1)
        logit_bin, logit_multi = self.heads(fused)
        return logit_bin, logit_multi

def multi_task_loss(logit_bin, logit_multi, label_bin, label_multi):
    loss_bin = F.binary_cross_entropy_with_logits(logit_bin, label_bin.float())
    loss_multi = F.cross_entropy(logit_multi, label_multi)
    return loss_bin + LAMBDA_ATTR * loss_multi, loss_bin.item(), loss_multi.item()

def load_dataset():
    class_map = [
        ("original", 0),
        ("synth_same_text", 1),
        ("synth_random_text", 2),
    ]
    all_paths = []
    all_labels = []
    for subdir, cls_id in class_map:
        folder = os.path.join("data", subdir)
        files = glob(os.path.join(folder, "*.wav"))
        all_paths.extend(files)
        all_labels.extend([cls_id]*len(files))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    return train_paths, val_paths, train_labels, val_labels

def train():
    logger.info("Loading dataset...")
    train_paths, val_paths, train_cls, val_cls = load_dataset()
    logger.info(f"Train size: {len(train_paths)} | Val size: {len(val_paths)}")

    train_dataset = MultiTaskDataset(train_paths, train_cls)
    val_dataset   = MultiTaskDataset(val_paths,   val_cls)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    model = PatentTTSNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    logger.info("Starting training...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_train_loss = 0.0
        total_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]", unit="batch")
        for mels, arts, lbl_bin, lbl_multi in train_pbar:
            mels = mels.to(DEVICE)
            arts = arts.to(DEVICE)
            lbl_bin = lbl_bin.to(DEVICE)
            lbl_multi = lbl_multi.to(DEVICE)

            logit_bin, logit_multi = model(mels, arts)
            loss, lb, lm = multi_task_loss(logit_bin, logit_multi, lbl_bin, lbl_multi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = mels.size(0)
            total_train_loss += loss.item() * bs
            total_samples += bs

            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "bin_loss": f"{lb:.4f}",
                "mc_loss": f"{lm:.4f}"
            })

        avg_loss = total_train_loss / total_samples

        model.eval()
        total_val = 0
        correct_bin = 0
        correct_multi = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [VAL]", unit="batch", leave=False)
        with torch.no_grad():
            for mels, arts, lbl_bin, lbl_multi in val_pbar:
                mels = mels.to(DEVICE)
                arts = arts.to(DEVICE)
                lbl_bin = lbl_bin.to(DEVICE)
                lbl_multi = lbl_multi.to(DEVICE)

                logit_bin, logit_multi = model(mels, arts)
                pred_bin = (torch.sigmoid(logit_bin) > 0.5).long()
                pred_multi = logit_multi.argmax(dim=1)

                correct_bin += (pred_bin == lbl_bin).sum().item()
                correct_multi += (pred_multi == lbl_multi).sum().item()
                total_val += lbl_bin.size(0)

        bin_acc = correct_bin / total_val
        mc_acc  = correct_multi / total_val

        logger.info(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val BinAcc: {bin_acc:.3f} | Val MCAcc: {mc_acc:.3f}"
        )

    os.makedirs("models", exist_ok=True)
    model_path = "models/patent_tts_net.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
