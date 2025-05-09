import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
from sklearn.model_selection import train_test_split
from glob import glob

SAMPLE_RATE = 22050
N_MELS = 80
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-4

def extract_mel(audio_path, sample_rate=SAMPLE_RATE, n_mels=N_MELS):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

class VoiceDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mel = extract_mel(self.file_paths[idx])
        mel = torch.tensor(mel).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx]).long()
        return mel, label

def pad_collate(batch):
    mels, labels = zip(*batch)
    max_len = max(m.shape[-1] for m in mels)
    padded_mels = []
    for m in mels:
        pad_size = max_len - m.shape[-1]
        padded = torch.nn.functional.pad(m, (0, pad_size))
        padded_mels.append(padded)
    return torch.stack(padded_mels), torch.tensor(labels)

class VoxTruthNet(nn.Module):
    def __init__(self):
        super(VoxTruthNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        self.pooling = nn.AdaptiveAvgPool2d((20, 20))
        self.fc = nn.Sequential(
            nn.Linear(32 * 20 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        attn = self.attention(x)
        x = x * attn
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_dataset():
    paths = []
    labels = []
    for i, subdir in enumerate(["original", "synth_same_text", "synth_random_text"]):
        subdir_path = os.path.join("data", subdir)
        files = glob(os.path.join(subdir_path, "*.wav"))
        paths.extend(files)
        labels.extend([i] * len(files))
    return train_test_split(paths, labels, test_size=0.2, random_state=42)

def train():
    train_paths, val_paths, train_labels, val_labels = load_dataset()
    train_dataset = VoiceDataset(train_paths, train_labels)
    val_dataset = VoiceDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate)

    model = VoxTruthNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for mel, label in train_loader:
            out = model(mel)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for mel, label in val_loader:
                out = model(mel)
                pred = out.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Val Acc: {correct/total:.2f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/voxtruthnet.pth")

if __name__ == "__main__":
    train()
