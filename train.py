import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.cuda.amp as amp

from transformers import BertModel, ViTModel
from transformers import BertTokenizer, ViTImageProcessor

# sklearn is only needed for evaluation metrics during training.
try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
except Exception:  # pragma: no cover
    accuracy_score = f1_score = roc_auc_score = None

from config import Config
import pickle

# Optional dependency; only needed if you call get_transform().
try:
    from torchvision import transforms
except Exception:  # pragma: no cover
    transforms = None

from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Local/offline HF assets
HF_LOCAL_DIR = Path(__file__).resolve().parent / "local_hf"
HF_VIT_DIR = HF_LOCAL_DIR / "vit-base"
HF_BERT_DIR = HF_LOCAL_DIR / "bert-base"

def prepare_offline_models(force=False):
    from hf_local import download_hf_models
    download_hf_models(force=force)

def _hf_path_or_id(local_path: Path, model_id: str) -> tuple[str, bool]:
    """Return (path_or_id, local_only)."""
    if local_path.exists() and any(local_path.iterdir()):
        return str(local_path), True
    return model_id, False


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vit_src, local_only = _hf_path_or_id(HF_VIT_DIR, "google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained(vit_src, local_files_only=local_only)

    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        return outputs.last_hidden_state  # [B, N, D]


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        bert_src, local_only = _hf_path_or_id(HF_BERT_DIR, "bert-base-uncased")
        self.bert = BertModel.from_pretrained(bert_src, local_files_only=local_only)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state  # [B, M, D]


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, text_feats, image_feats):
        fused, _ = self.attn(
            query=text_feats,
            key=image_feats,
            value=image_feats
        )
        return fused


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class CrossModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.cross_attn = CrossAttention(hidden_dim=768)
        self.classifier = ClassificationHead(hidden_dim=768)


    @property
    def tokenizer(self):
        if self._tokenizer is None:
            bert_src, local_only = _hf_path_or_id(HF_BERT_DIR, "bert-base-uncased")
            self._tokenizer = BertTokenizer.from_pretrained(bert_src, local_files_only=local_only)
        return self._tokenizer

    @property
    def image_processor(self):
        if self._image_processor is None:
            vit_src, local_only = _hf_path_or_id(HF_VIT_DIR, "google/vit-base-patch16-224-in21k")
            self._image_processor = ViTImageProcessor.from_pretrained(vit_src, local_files_only=local_only)
        return self._image_processor

    def encode_text(self, text: str, *, device=None):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=Config.max_text_len,
            return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def encode_image(self, image, *, device=None):
        # image: PIL.Image or numpy array; ViTImageProcessor supports both.
        out = self.image_processor(image.convert("RGB"), return_tensors="pt")["pixel_values"]
        if device is not None:
            out = out.to(device)
        return out

    def forward(self, images, input_ids, attention_mask):
        img_feats = self.image_encoder(images)
        txt_feats = self.text_encoder(input_ids, attention_mask)

        fused = self.cross_attn(txt_feats, img_feats)

        cls_token = fused[:, 0]  # CLS
        logits = self.classifier(cls_token)

        return logits

    @torch.no_grad()
    def predict_proba(self, image, text: str, *, device=None) -> float:
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        pixel_values = self.encode_image(image, device=device)
        enc = self.encode_text(text, device=device)

        logits = self(pixel_values, enc["input_ids"], enc["attention_mask"])
        return float(torch.sigmoid(logits).item())

    @torch.no_grad()
    def predict(self, image, text: str, *, threshold: float = 0.5, device=None) -> int:
        p = self.predict_proba(image, text, device=device)
        return int(p >= threshold)

    def get_transform(self):
        return get_transform()


def get_transform():
    """Standalone image transform matching the ViT preprocessing used by this project.

    Uses the same (possibly offline-loaded) `ViTImageProcessor` config to construct a
    torchvision transform: Resize -> ToTensor -> Normalize.

    Raises:
        ImportError: if torchvision isn't installed.
    """
    if transforms is None:
        raise ImportError(
            "torchvision is required for get_transform(). Install it or use ViTImageProcessor directly."
        )

    vit_src, local_only = _hf_path_or_id(HF_VIT_DIR, "google/vit-base-patch16-224-in21k")
    p = ViTImageProcessor.from_pretrained(vit_src, local_files_only=local_only)

    size = getattr(p, "size", None)
    if isinstance(size, dict):
        height = int(size.get("height", 224))
        width = int(size.get("width", 224))
    elif isinstance(size, (int, float)):
        height = width = int(size)
    else:
        height = width = 224

    mean = list(getattr(p, "image_mean", [0.5, 0.5, 0.5]))
    std = list(getattr(p, "image_std", [0.5, 0.5, 0.5]))

    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    losses = []

    for batch in tqdm(loader):
        optimizer.zero_grad()

        with amp.autocast():
            logits = model(
                batch["image"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )

            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch["label"].to(device)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return np.mean(losses)


@torch.no_grad()
def evaluate(model, loader):
    if accuracy_score is None or f1_score is None or roc_auc_score is None:
        raise ImportError(
            "scikit-learn (and its dependencies) is required for evaluate(). "
            "Install it or skip evaluation metrics."
        )

    model.eval()

    preds, labels = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        logits = model(
            batch["image"].to(device),
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )

        probs = torch.sigmoid(logits)
        preds.extend(probs.cpu().numpy())
        labels.extend(batch["label"].numpy())

    preds_bin = (np.array(preds) > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds_bin),
        "f1": f1_score(labels, preds_bin),
        "auc": roc_auc_score(labels, preds)
    }



def train():
    print("Loading dataset from dataset.pkl...")
    with open("dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print(f"Loaded dataset with {len(dataset)} samples.")

    print("Splitting dataset into train and validation sets...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train set: {train_size} samples, Validation set: {val_size} samples.")

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    print(f"Train loader: {len(train_loader)} batches, Validation loader: {len(val_loader)} batches.")

    print("Initializing model and optimizer...")
    model = CrossModalModel().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.image_encoder.parameters(), "lr": Config.lr_backbone},
        {"params": model.text_encoder.parameters(), "lr": Config.lr_backbone},
        {"params": model.classifier.parameters(), "lr": Config.lr_head},
    ])
    scaler = amp.GradScaler()
    print("Model, optimizer, and GradScaler initialized. Using mixed precision (float16).")

    best_auc = 0.0
    print(f"Starting training for {Config.num_epochs} epochs...")

    for epoch in tqdm(range(Config.num_epochs), desc="Training Epochs"):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        print(f"Train Loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader)
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), "weights.pth")
            print("Saved best model (weights.pth)")

    print("Training complete. Best AUC:", best_auc)


if __name__ == "__main__":
    train()