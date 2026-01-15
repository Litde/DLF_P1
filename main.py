import os
import random
import csv
from collections import defaultdict

import torch
from PIL import Image

from train import CrossModalModel, ImageEncoder, TextEncoder, CrossAttention, ClassificationHead


def load_captions_txt(path: str) -> dict[str, list[str]]:
    """Load Flickr-style captions file: each row `image.jpg,caption`.

    Returns dict: image_id_without_ext -> [caption1, caption2, ...]
    """
    caps: defaultdict[str, list[str]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            img_name = row[0].strip()
            caption = row[1].strip()
            if not img_name or not caption:
                continue
            img_id = os.path.splitext(os.path.basename(img_name))[0]
            caps[img_id].append(caption)
    return dict(caps)


def pick_random_image(images_dir: str) -> str:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    files = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f)) and os.path.splitext(f)[1].lower() in exts
    ]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return random.choice(files)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    state_dict = torch.load("main_model.pt", map_location=device)
    model = CrossModalModel().to(device)
    model.load_state_dict(state_dict)

    # Pick random image
    images_dir = os.path.join("archive", "Images")
    captions_path = os.path.join("archive", "captions.txt")

    if not os.path.isfile(captions_path):
        raise FileNotFoundError(f"Missing captions file: {captions_path}")

    captions = load_captions_txt(captions_path)

    image_path = pick_random_image(images_dir)
    image_id = os.path.splitext(os.path.basename(image_path))[0]

    # Pick a caption for this image (fallback to any caption if missing)
    if image_id in captions and captions[image_id]:
        caption = random.choice(captions[image_id])
        caption_source = "matching-caption"
    else:
        any_ids = [k for k, v in captions.items() if v]
        if not any_ids:
            raise RuntimeError("captions.txt parsed, but no captions found.")
        fallback_id = random.choice(any_ids)
        caption = random.choice(captions[fallback_id])
        caption_source = f"random-caption-from-{fallback_id}"

    image = Image.open(image_path)
    prob = model.predict(image, caption, device=device)
    pred = 1 if prob >= 0.5 else 0

    print("Single-sample test")
    print(f"  device: {device}")
    print(f"  image: {image_path}")
    print(f"  caption_source: {caption_source}")
    print(f"  caption: {caption}")
    print(f"  match_probability(sigmoid): {prob:.4f}")
    print(f"  predicted_label(threshold=0.5): {pred}")

def tmp_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    state_dict = torch.load("main_model.pt", map_location=device)
    model = CrossModalModel().to(device)
    model.load_state_dict(state_dict)

    torch.save(model.state_dict(), "weights.pth")


if __name__ == "__main__":
    tmp_save()
