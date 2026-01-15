import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.cuda.amp as amp

from transformers import BertModel
from transformers import ViTModel

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import Config
from prepare_dataset import CrossModalDataset
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        return outputs.last_hidden_state  # [B, N, D]


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

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

    def forward(self, images, input_ids, attention_mask):
        img_feats = self.image_encoder(images)
        txt_feats = self.text_encoder(input_ids, attention_mask)

        fused = self.cross_attn(txt_feats, img_feats)

        cls_token = fused[:, 0]  # CLS
        logits = self.classifier(cls_token)

        return logits


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
            torch.save(model.state_dict(), "best_cross_modal_model.pt")
            print("Saved best model")

    print("Training complete. Best AUC:", best_auc)


if __name__ == "__main__":
    train()