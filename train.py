import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 16000       # Частота дискретизации
N_MELS = 80               # Число мел-бинов
WIN_LENGTH = 400          # Размер окна STFT ~ 25 мс (при 16 кГц)
HOP_LENGTH = 160          # Шаг окна STFT ~ 10 мс
FMIN = 50                 # Мин. частота мелового банка
FMAX = 8000               # Макс. частота мелового банка
D_MODEL = 256             # Размерность эмбеддингов в Трансформере
NUM_HEADS = 4             # Число голов внимания
NUM_LAYERS = 4            # Число слоёв Transformer Encoder
DIM_FEEDFORWARD = 512     # Размер скрытого слоя в FFN
PATCH_TIME = 4            # «Высота» патча по временной оси (кол-во фреймов)
PATCH_FREQ = 4            # «Ширина» патча по частотной оси (кол-во мел-бинов)
SEGMENT_SIZE = 16         # Число патчей в одном сегменте для «вторичного» self-attn
NUM_SEGMENT_LAYERS = 2    # Число слоёв «сегментного» self-attn

# Для мультизадачности:
NUM_SYNTH_CLASSES = 4     # Пример: K=4 (4 вида TTS) + 1 реальный (опционально)
LAMBDA_ATTR = 1.0         # Вес лосса атрибуции

# Параметры обучения (примерные)
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomAudioDataset(Dataset):
    def __init__(self, length=10):
        super().__init__()
        self.length = length
        self.labels_bin = []
        self.labels_multi = []
        for i in range(length):
            is_fake = torch.randint(0, 2, (1,)).item()
            if is_fake == 0:
                synth_label = 0  # Real
            else:
                synth_label = torch.randint(1, NUM_SYNTH_CLASSES, (1,)).item()
            self.labels_bin.append(is_fake)
            self.labels_multi.append(synth_label)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        audio = torch.randn(1, SAMPLE_RATE)

        label_bin = self.labels_bin[idx]
        label_multi = self.labels_multi[idx]

        return audio, label_bin, label_multi


def compute_mel_spectrogram(audio):
    """
    Вычисляем лог-мел-спектрограмму.
    audio.shape = (1, num_samples). Возвращаем тензор (Time, Mel).
    """
    # Создаём MelSpectrogram из torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        f_min=FMIN,
        f_max=FMAX,
        power=2.0
    )
    # Вычисляем мел-спектрограмму
    mel = mel_transform(audio)  # (1, N_MELS, Time)
    mel_db = torch.log(mel.clamp(min=1e-5))  # логарифм (можно дБ)
    mel_db = mel_db.squeeze(0).transpose(0, 1)  # -> (Time, Mel)
    return mel_db


def compute_delta_features(mel_spec):
    """
    Дополнительно считаем дельта и дельта-дельта.
    Возвращаем тензор (Time, Mel, 3), где:
      [:,:,0] - статические признаки,
      [:,:,1] - дельта,
      [:,:,2] - дельта-дельта.
    """
    delta = torchaudio.functional.compute_deltas(mel_spec.transpose(0,1)).transpose(0,1)
    delta2 = torchaudio.functional.compute_deltas(delta.transpose(0,1)).transpose(0,1)
    mel_stack = torch.stack([mel_spec, delta, delta2], dim=-1)  # (Time, Mel, 3)
    return mel_stack


def compute_artifact_map(mel_spec, kernel_size=9):

    padding = kernel_size // 2
    smooth = F.avg_pool1d(
        mel_spec.unsqueeze(1).transpose(1,2), 
        kernel_size, 
        stride=1, 
        padding=padding
    ).transpose(1,2).squeeze(1)
    artifact = mel_spec - smooth
    return artifact


def collate_fn(batch):

    audios, labels_bin, labels_multi = zip(*batch)
    mel_list = []
    artifact_list = []
    for audio in audios:
        mel = compute_mel_spectrogram(audio)
        # Вычислим дельта-признаки
        mel_stack = compute_delta_features(mel)
        # Вычислим карту артефактов (доп. канал)
        art = compute_artifact_map(mel)
        
        mel_list.append(mel_stack)
        artifact_list.append(art)
    
    max_len = max(mel.shape[0] for mel in mel_list)

    def pad_mel_stack(mel_stack, max_len):
        time_len, n_mels, ch = mel_stack.shape
        pad_len = max_len - time_len
        if pad_len > 0:
            padding = (0,0, 0,0, 0,pad_len)  # (dim=-1, -2, -3)
            mel_stack = F.pad(mel_stack, padding)
        return mel_stack

    def pad_artifact(art, max_len):
        time_len, n_mels = art.shape
        pad_len = max_len - time_len
        if pad_len > 0:
            padding = (0,0, 0,pad_len)
            art = F.pad(art.transpose(0,1), padding).transpose(0,1)
        return art

    mel_padded = [pad_mel_stack(m, max_len) for m in mel_list]
    art_padded = [pad_artifact(a, max_len) for a in artifact_list]

    mel_input = torch.stack(mel_padded, dim=0)        # [B, Time, Mel, 3]
    artifact_input = torch.stack(art_padded, dim=0)   # [B, Time, Mel]

    labels_bin = torch.tensor(labels_bin, dtype=torch.long)
    labels_multi = torch.tensor(labels_multi, dtype=torch.long)

    return mel_input, artifact_input, labels_bin, labels_multi


class PatchEmbed(nn.Module):

    def __init__(self, d_model, patch_time, patch_freq, in_channels=3):
        super().__init__()
        self.d_model = d_model
        self.patch_time = patch_time
        self.patch_freq = patch_freq
        self.in_channels = in_channels
        
        # Размерность входного вектора на один патч:
        # patch_time * patch_freq * in_channels
        self.patch_dim = patch_time * patch_freq * in_channels
        
        self.proj = nn.Linear(self.patch_dim, d_model)

    def forward(self, x):
        """
        x: [B, Time, Mel, Channels]
        """
        B, T, F, C = x.shape
        
        new_T = (T // self.patch_time) * self.patch_time
        new_F = (F // self.patch_freq) * self.patch_freq
        x = x[:, :new_T, :new_F, :]

        # Теперь разбиваем на патчи (batch, nT, patch_time, nF, patch_freq, channels)
        nT = new_T // self.patch_time
        nF = new_F // self.patch_freq
        x = x.view(B, nT, self.patch_time, nF, self.patch_freq, C)
        
        # Переставим оси, чтобы собрать (B, nT*nF, patch_time*patch_freq*C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, nT, nF, patch_time, patch_freq, C]
        x = x.view(B, nT * nF, self.patch_dim)        # [B, N, patch_dim], где N = nT*nF
        
        # Прогоним через линейный слой
        x = self.proj(x)  # [B, N, d_model]
        return x


class ArtifactEmbed(nn.Module):

    def __init__(self, d_model, patch_time, patch_freq):
        super().__init__()
        self.d_model = d_model
        self.patch_time = patch_time
        self.patch_freq = patch_freq
        
        self.patch_dim = patch_time * patch_freq  # т.к. 1 канал
        self.proj = nn.Linear(self.patch_dim, d_model)

    def forward(self, art):
        """
        art: [B, Time, Mel] (1 канал).
        Возвращает [B, N, d_model].
        """
        B, T, F = art.shape
        new_T = (T // self.patch_time) * self.patch_time
        new_F = (F // self.patch_freq) * self.patch_freq
        art = art[:, :new_T, :new_F]

        nT = new_T // self.patch_time
        nF = new_F // self.patch_freq

        art = art.view(B, nT, self.patch_time, nF, self.patch_freq)  # [B, nT, patch_time, nF, patch_freq]
        art = art.permute(0, 1, 3, 2, 4).contiguous()                # [B, nT, nF, patch_time, patch_freq]
        art = art.view(B, nT * nF, self.patch_dim)                   # [B, N, patch_dim]

        art = self.proj(art)  # [B, N, d_model]
        return art

class TransformerEncoderWrapper(nn.Module):

    def __init__(self, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # [CLS] - параметр
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Позиционные эмбеддинги (для упрощения - обучаемые)
        self.pos_embed = nn.Parameter(torch.zeros(1, 10000, d_model))  # max 10000 патчей, MVP
        
    def forward(self, x, x_art):
        """
        x:     [B, N_main, d_model]  -- патч-эмбеддинги основной спектрограммы
        x_art: [B, N_art,  d_model]  -- патч-эмбеддинги артефактной карты
        """
        B, N_main, _ = x.shape
        B, N_art,  _ = x_art.shape
        
        # Соединим последовательности: [CLS] + x + x_art
        cls_token = self.cls_token.repeat(B, 1, 1)  # [B, 1, d_model]
        x_all = torch.cat([cls_token, x, x_art], dim=1)  # [B, 1+N_main+N_art, d_model]
        
        positions = self.pos_embed[:, : (1 + N_main + N_art), :]  # [1, seq_len, d_model]
        x_all = x_all + positions
        
        # Прогоняем через энкодер
        x_enc = self.transformer(x_all)  # [B, seq_len, d_model]
        
        # Выделим [CLS]
        cls_emb = x_enc[:, 0, :]  # [B, d_model]
        
        enc_tokens = x_enc[:, 1:, :]  # [B, N_main+N_art, d_model] (без CLS)
        
        return cls_emb, enc_tokens

class SegmentLevelSelfAttention(nn.Module):

    def __init__(self, d_model, nhead=2, num_layers=2, dim_feedforward=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.segment_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, enc_tokens):
        """
        enc_tokens: [B, N, d_model]
        """
        B, N, D = enc_tokens.shape
        nSeg = N // SEGMENT_SIZE
        used_len = nSeg * SEGMENT_SIZE
        tokens = enc_tokens[:, :used_len, :]
        tokens = tokens.view(B, nSeg, SEGMENT_SIZE, D)  # [B, nSeg, segSize, d_model]
        
        seg_emb = tokens.mean(dim=2)  # [B, nSeg, d_model]
        
        seg_emb_enc = self.segment_encoder(seg_emb)  # [B, nSeg, d_model]
        
        seg_rep = seg_emb_enc.mean(dim=1)  # [B, d_model] (агрегация по сегментам)
        
        return seg_rep


class ClassificationHeads(nn.Module):
    """
    - Детекция (бинарная классификация): Real vs Fake
    - Атрибуция (многоклассовая классификация): какой TTS (K классов)
    """
    def __init__(self, d_model, num_synth_classes):
        super().__init__()
        # Бинарный классификатор
        self.bin_fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)  # будем выдавать логит 1 шт
        )
        # Мультиклассовый классификатор
        self.multi_fc = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, num_synth_classes)  # K логитов
        )
    
    def forward(self, x):
        """
        x: [B, d_model] -- свёрнутое представление (после всех модулей)
        """
        logit_bin = self.bin_fc(x).squeeze(-1)           # [B], логит для сигмоида (или 2 выхода, если угодно)
        logit_multi = self.multi_fc(x)                   # [B, K]
        return logit_bin, logit_multi


class VoiceSpoofTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Patch-Embedding
        self.patch_embed_main = PatchEmbed(
            d_model=D_MODEL,
            patch_time=PATCH_TIME,
            patch_freq=PATCH_FREQ,
            in_channels=3  # статические + дельта + дельта^2
        )
        # Artifact-Embedding
        self.patch_embed_art = ArtifactEmbed(
            d_model=D_MODEL,
            patch_time=PATCH_TIME,
            patch_freq=PATCH_FREQ
        )
        
        # Трансформерный энкодер
        self.encoder = TransformerEncoderWrapper(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD
        )
        
        # Сегментный анализ
        self.segment_analyzer = SegmentLevelSelfAttention(
            d_model=D_MODEL,
            nhead=2,
            num_layers=NUM_SEGMENT_LAYERS,
            dim_feedforward=256
        )
        
        # Итоговые головы
        self.class_heads = ClassificationHeads(D_MODEL*2, NUM_SYNTH_CLASSES)


    def forward(self, mel_input, artifact_input):
        """
        mel_input:      [B, Time, Mel, 3]
        artifact_input: [B, Time, Mel]
        """
        # 1) Эмбеддинги патчей
        x_main = self.patch_embed_main(mel_input)     # [B, N_main, d_model]
        x_art  = self.patch_embed_art(artifact_input) # [B, N_art, d_model]
        
        # 2) Трансформер
        cls_emb, enc_tokens = self.encoder(x_main, x_art)  # cls_emb: [B, d_model], enc_tokens: [B, N_main+N_art, d_model]
        
        # 3) Сегментный анализ
        seg_rep = self.segment_analyzer(enc_tokens)   # [B, d_model]
        
        # 4) Конкатенируем cls_emb и seg_rep
        fused_rep = torch.cat([cls_emb, seg_rep], dim=-1)  # [B, 2*d_model]
        
        # 5) Две головы
        logit_bin, logit_multi = self.class_heads(fused_rep)  # [B], [B, K]
        
        return logit_bin, logit_multi


def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    
    total_loss = 0.0
    total_bin_acc = 0
    total_count = 0

    for mel_input, artifact_input, label_bin, label_multi in loader:
        mel_input = mel_input.to(DEVICE)            # [B, T, F, 3]
        artifact_input = artifact_input.to(DEVICE)  # [B, T, F]
        label_bin = label_bin.to(DEVICE)            # [B]
        label_multi = label_multi.to(DEVICE)        # [B]

        optimizer.zero_grad()
        logit_bin, logit_multi = model(mel_input, artifact_input)

        loss_bin = F.binary_cross_entropy_with_logits(logit_bin, label_bin.float())

        loss_multi = F.cross_entropy(logit_multi, label_multi)

        loss = loss_bin + LAMBDA_ATTR * loss_multi
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * label_bin.size(0)

        preds_bin = (torch.sigmoid(logit_bin) > 0.5).long()
        correct_bin = (preds_bin == label_bin).sum().item()
        total_bin_acc += correct_bin
        total_count += label_bin.size(0)
    
    avg_loss = total_loss / total_count
    avg_acc = total_bin_acc / total_count
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Bin Acc: {avg_acc:.4f}")


def main():

    dataset = RandomAudioDataset(length=10)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = VoiceSpoofTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        train_one_epoch(model, loader, optimizer, epoch)


if __name__ == "__main__":
    main()
