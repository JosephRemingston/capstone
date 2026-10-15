"""The frozen-BGE deep head (training only; serving runs its ONNX export through onnxruntime)."""
from __future__ import annotations

from main.ml import TEXT_FEATURES

ARCHITECTURE = "frozen-bge-deep-head-v1"


def create_head():
    """Return a trainable deep head: text 384→192→128, numeric 13→64→32, joint 160→128→64→3."""
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Install requirements.txt (torch) to train the importance model") from exc

    class RetentionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_branch = torch.nn.Sequential(
                torch.nn.Linear(384, 192), torch.nn.LayerNorm(192),
                torch.nn.GELU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(192, 128), torch.nn.LayerNorm(128),
                torch.nn.GELU(), torch.nn.Dropout(0.2),
            )
            self.numeric_branch = torch.nn.Sequential(
                torch.nn.Linear(len(TEXT_FEATURES), 64), torch.nn.LayerNorm(64),
                torch.nn.GELU(), torch.nn.Linear(64, 32),
                torch.nn.LayerNorm(32), torch.nn.GELU(),
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(160, 128), torch.nn.GELU(),
                torch.nn.Dropout(0.2), torch.nn.Linear(128, 64),
                torch.nn.GELU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 3),
            )

        def forward(self, embedding, numeric):
            if embedding.ndim != 2 or embedding.shape[1] != 384:
                raise ValueError("Expected 384-dimensional text embeddings")
            if numeric.ndim != 2 or numeric.shape[1] != len(TEXT_FEATURES):
                raise ValueError("Expected 13 numeric features")
            combined = torch.cat((self.text_branch(embedding),
                                  self.numeric_branch(numeric)), dim=1)
            return self.head(combined)

    return RetentionHead()
