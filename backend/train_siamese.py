import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import PaperSignatureCNN
from preprocessing import full_preprocessing

class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese metric learning."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

class CedarSiameseDataset(Dataset):
    """
    Pairs signatures from the CEDAR dataset:
      - Positive Pair (label 0): Two genuine signatures from writer i
      - Negative Pair (label 1): One genuine signature & one forgery of writer i
    """
    def __init__(self, cedar_dir):
        self.org_dir = os.path.join(cedar_dir, "full_org")
        self.forg_dir = os.path.join(cedar_dir, "full_forg")
        
        # Organize images by writer ID (1 to 55)
        self.genuine_dict = {}
        self.forgery_dict = {}

        for path in glob.glob(os.path.join(self.org_dir, "original_*_*.png")):
            fname = os.path.basename(path)
            parts = fname.replace(".png", "").split("_")
            writer_id = int(parts[1])
            self.genuine_dict.setdefault(writer_id, []).append(path)

        for path in glob.glob(os.path.join(self.forg_dir, "forgeries_*_*.png")):
            fname = os.path.basename(path)
            parts = fname.replace(".png", "").split("_")
            writer_id = int(parts[1])
            self.forgery_dict.setdefault(writer_id, []).append(path)

        self.writers = list(self.genuine_dict.keys())
        self.pairs = []
        self._generate_pairs()

    def _generate_pairs(self):
        # Build balanced 50/50 dataset of genuine-genuine and genuine-forgery pairs
        for writer in self.writers:
            genuines = self.genuine_dict.get(writer, [])
            forgeries = self.forgery_dict.get(writer, [])

            # Positive Pairs (Genuine vs Genuine)
            for i in range(len(genuines)):
                for j in range(i + 1, min(i + 4, len(genuines))):
                    self.pairs.append((genuines[i], genuines[j], 0.0))

            # Negative Pairs (Genuine vs Forgery of same writer)
            if forgeries:
                for i in range(min(len(genuines), 10)):
                    forg_sample = random.choice(forgeries)
                    self.pairs.append((genuines[i], forg_sample, 1.0))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _load_and_preprocess(self, path):
        with open(path, "rb") as f:
            img_bytes = f.read()
        _, cnn_input = full_preprocessing(img_bytes, target_size=(64, 64))
        tensor = torch.from_numpy(cnn_input).float().unsqueeze(0) / 255.0
        return tensor

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        t1 = self._load_and_preprocess(path1)
        t2 = self._load_and_preprocess(path2)
        return t1, t2, torch.tensor(label, dtype=torch.float32)

def train_siamese(cedar_dir="../datasets/CEDAR", epochs=20, batch_size=32, lr=0.0005):
    if not os.path.exists(cedar_dir):
        print(f"Error: Path {cedar_dir} does not exist.")
        return

    dataset = CedarSiameseDataset(cedar_dir)
    print(f"Constructed {len(dataset)} Siamese verification pairs across {len(dataset.writers)} signers.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PaperSignatureCNN().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            emb1 = model.get_embedding(img1)
            emb2 = model.get_embedding(img2)
            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img1.size(0)

        epoch_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Contrastive Loss: {epoch_loss:.5f}")

    weights_out = "signature_cnn.pth"
    torch.save(model.state_dict(), weights_out)
    print(f"Saved Siamese verification weights to '{weights_out}'.")

if __name__ == "__main__":
    train_siamese()
