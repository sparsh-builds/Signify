import torch
import torch.nn as nn
import torch.nn.functional as F

class PaperSignatureCNN(nn.Module):
    """
    Siamese CNN architecture for offline signature feature extraction.
    Takes 1x64x64 preprocessed stroke inputs and projects them into
    a 128-dimensional invariant latent embedding space.
    """
    def __init__(self, num_classes: int = 50):
        super(PaperSignatureCNN, self).__init__()
        
        # Feature Extractor (Backbone)
        self.features = nn.Sequential(
            # Block 1: 1 x 64 x 64 -> 32 x 32 x 32
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 x 8 x 8 -> 256 x 4 x 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 128-d Latent Embedding Projection Head
        self.fc_embed = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128)
        )

        # Classification Head (used during training on CEDAR/BHSig benchmarks)
        self.classifier = nn.Linear(128, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts L2-normalized 128-dimensional latent vector for zero-shot
        cosine similarity evaluation.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feat = self.fc_embed(x)
        # Unit normalization for direct dot-product cosine similarity
        norm_feat = F.normalize(feat, p=2, dim=1)
        return norm_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.get_embedding(x)
        out = self.classifier(feat)
        return out
    