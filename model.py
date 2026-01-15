"""
SUBMISSION TEMPLATE FOR IMAGE-TEXT MATCHING COMPETITION

Instructions:
1. Implement your model in the SubmissionModel class
2. Optionally customize get_transform() for your preprocessing
3. Save weights: torch.save(model.state_dict(), 'weights.pth')
4. Create submission: zip submission.zip model.py weights.pth
5. Upload to: /submissions/queue/YOUR_TEAM_NAME/submission.zip

Task: Predict if an image and text caption match (1) or not (0)
Output: Float between 0.0 and 1.0 (≥0.5 = match)
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import BertModel, ViTModel
from transformers import BertTokenizer, ViTImageProcessor
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

threshold: float = 0.5

# Local/offline HF assets
HF_LOCAL_DIR = Path(__file__).resolve().parent / "local_hf"
HF_VIT_DIR = HF_LOCAL_DIR / "vit-base"
HF_BERT_DIR = HF_LOCAL_DIR / "bert-base"


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


def get_transform():
    """
    OPTIONAL: Define custom image preprocessing.
    If omitted, default transform will be used (Resize 224, ImageNet normalization).

    Returns:
        torchvision.transforms.Compose: Preprocessing pipeline

    Example with augmentation:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    """

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

class SubmissionModel(nn.Module):
    """
    REQUIRED: Main model class for image-text matching.

    Must implement:
        - __init__(): Initialize your model
        - predict(image_tensor, text_string): Inference method
    """

    def __init__(self):
        """
        Initialize your model architecture here.

        Tips:
        - Here you can use pretrained tokenizers/encoders/models
        - THERE IS NO INTERNET ACCESS DURING EVALUATION, load all resources LOCALLY
        - Keep model size reasonable (CPU RAM limit: 32GB, GPU VRAM limit: 20 GB)
        - Optimize for inference speed (10 min timeout)
        """
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.cross_attn = CrossAttention(hidden_dim=768)
        self.classifier = ClassificationHead(hidden_dim=768)

        self._tokenizer = None
        self._image_processor = None

        # Place modules on the same device used by evaluator.
        self.to(device)

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

    def encode_text(self, text, *, device=None):
        """
        Helper method: Convert text to features.

        Here you can implement proper text encoding:

        """
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

    def prepare_image_tensor(self, image_tensor: torch.Tensor, *, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"image_tensor must be a torch.Tensor, got {type(image_tensor)}")

        x = image_tensor
        # Move to device early so all ops are consistent.
        x = x.to(device)

        # Ensure float
        if x.dtype != torch.float32:
            x = x.float()

        # Resize if needed (assume CHW)
        # Use torchvision resize on tensor if available. If not, rely on ViT accepting 224x224 only.
        # We mirror the processor size.
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

        # If shape isn't correct, resize using interpolate
        if x.shape[1] != height or x.shape[2] != width:
            x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)

        mean = torch.tensor(
            getattr(p, "image_mean", [0.5, 0.5, 0.5]),
            dtype=torch.float32,
            device=device,
        ).view(3, 1, 1)
        std = torch.tensor(
            getattr(p, "image_std", [0.5, 0.5, 0.5]),
            dtype=torch.float32,
            device=device,
        ).view(3, 1, 1)

        x = (x - mean) / std

        return x.unsqueeze(0)

    def forward(self, images, input_ids, attention_mask):
        """Forward pass compatible with the training model.

        Args:
            images: pixel_values tensor (B, 3, H, W)
            input_ids: token ids (B, L)
            attention_mask: mask (B, L)

        Returns:
            logits: (B,)
        """
        img_feats = self.image_encoder(images)
        txt_feats = self.text_encoder(input_ids, attention_mask)

        fused = self.cross_attn(txt_feats, img_feats)
        cls_token = fused[:, 0]
        logits = self.classifier(cls_token)
        return logits

    @torch.no_grad()
    def predict(self, image_tensor, text_string):
        """Return a Python float score in [0,1] as required by the evaluator."""
        # Always use the model's current device.
        dev = next(self.parameters()).device
        score = self.predict_proba(image_tensor, text_string, device=dev)
        return float(score)

    @torch.no_grad()
    def predict_proba(self, image_tensor, text: str, *, device=None) -> float:
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        pixel_values = self.prepare_image_tensor(image_tensor, device=device)
        enc = self.encode_text(text, device=device)

        logits = self(pixel_values, enc["input_ids"], enc["attention_mask"])
        return float(torch.sigmoid(logits).item())



# =============================================================================
# TESTING YOUR SUBMISSION LOCALLY
# =============================================================================

def test_submission():
    """
    Test your model before submitting.
    Run: python model.py
    """
    print("Testing submission...")

    # Create model
    model = SubmissionModel()
    model.eval()

    # Create dummy input
    dummy_image = torch.randn(3, 224, 224)
    dummy_text = "A dog running through a field"

    # Test predict method
    try:
        score = model.predict(dummy_image, dummy_text)
        print(f"✓ predict() works! Score: {score}")

        # Validate output
        if not isinstance(score, float):
            print(f"❌ ERROR: predict() must return float, got {type(score)}")
        elif not (0.0 <= score <= 1.0):
            print(f"❌ ERROR: Score {score} not in [0.0, 1.0]")
        else:
            print("✓ Output format valid")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Test transform
    try:
        transform = get_transform()
        from PIL import Image
        dummy_pil = Image.new('RGB', (256, 256))
        transformed = transform(dummy_pil)
        print(f"✓ get_transform() works! Output shape: {transformed.shape}")
    except Exception as e:
        print(f"⚠ get_transform() error: {e}")
        print("  (Will use default transform)")

    print("\n" + "=" * 60)
    print("SUBMISSION CHECKLIST:")
    print("=" * 60)
    print("[ ] Model architecture implemented")
    print("[ ] predict() method works correctly")
    print("[ ] Weights trained and saved to weights.pth")
    print("[ ] get_transform() defined (optional)")
    print("[ ] Tested locally with sample data")
    print("[ ] Created zip: zip submission.zip model.py weights.pth")
    print("[ ] Ready to upload!")
    print("=" * 60)


if __name__ == "__main__":
    test_submission()