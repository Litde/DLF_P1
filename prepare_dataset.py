from torch.utils.data import Dataset
from PIL import Image
import torch
from config import Config
import random
import pickle
from transformers import BertTokenizer, BertModel
from transformers import ViTModel, ViTImageProcessor
import os
import re
import csv
from collections import defaultdict

class CrossModalDataset(Dataset):
    def __init__(self, samples, tokenizer, image_processor):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Text
        encoding = self.tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=Config.max_text_len,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.float)
        }


def generate_negative_caption(caption: str) -> str:
    """
    Perform simple attribute swaps to create a negative caption:
    - color swaps (red <-> blue, black <-> white, etc.)
    - number swaps (one <-> two, three <-> four)
    - relation swaps (left <-> right)
    Fallback: swap two words or append 'different' if no substitution was made.
    """
    subs = {
        r"\bred\b": "blue",
        r"\bblue\b": "red",
        r"\bblack\b": "white",
        r"\bwhite\b": "black",
        r"\bgreen\b": "yellow",
        r"\bone\b": "two",
        r"\btwo\b": "one",
        r"\bthree\b": "four",
        r"\bfour\b": "three",
        r"\bleft\b": "right",
        r"\bright\b": "left",
        r"\bsmall\b": "large",
        r"\blarge\b": "small",
    }

    new_caption = caption
    changed = False
    for pattern, replacement in subs.items():
        if re.search(pattern, new_caption, flags=re.IGNORECASE):
            new_caption = re.sub(pattern, replacement, new_caption, flags=re.IGNORECASE)
            changed = True

    if not changed:
        words = new_caption.split()
        if len(words) >= 2:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
            new_caption = " ".join(words)
        else:
            new_caption = new_caption + " different"

    return new_caption


def build_dataset(positive_samples):
    """
    Accepts either:
    - a list of dicts with keys: 'image_path' (or 'path'/'img') and 'caption' (or 'text')
    - a list of image file paths (strings). In that case caption is derived from filename.
    Returns a shuffled list of samples with positive and corresponding negative examples.
    """
    samples = []
    for s in positive_samples:
        if isinstance(s, str):
            image_path = s
            caption = os.path.splitext(os.path.basename(s))[0].replace("_", " ")
        elif isinstance(s, dict):
            image_path = s.get("image_path") or s.get("path") or s.get("img")
            caption = s.get("caption") or s.get("text") or ""
            if image_path is None:
                continue
        else:
            continue

        if not os.path.isfile(image_path):
            # skip missing files
            continue

        samples.append({
            "image_path": image_path,
            "text": caption,
            "label": 1
        })

        neg_caption = generate_negative_caption(caption)
        samples.append({
            "image_path": image_path,
            "text": neg_caption,
            "label": 0
        })

    random.shuffle(samples)
    return samples

def prepare_samples(image_paths, captions_dict):
    """
    Connects image paths with multiple captions.
    Args:
        image_paths: List of image file paths.
        captions_dict: Dict where keys are image names (e.g., 'image1') and values are lists of captions.
    Returns:
        List of dicts with 'image_path' and 'text'.
    """
    samples = []
    for path in image_paths:
        image_name = os.path.splitext(os.path.basename(path))[0]
        if image_name in captions_dict:
            for caption in captions_dict[image_name]:
                samples.append({
                    "image_path": path,
                    "text": caption
                })
    return samples


def load_captions_from_file(filepath):
    """
    Load captions from a CSV text file.
    Assumes format: each line "image_name,caption"
    Returns a dict where keys are image names (without extension) and values are lists of captions.
    """
    captions_dict = defaultdict(list)
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    image_name = os.path.splitext(row[0].strip())[0]
                    caption = row[1].strip()
                    captions_dict[image_name].append(caption)
    return dict(captions_dict)


def main():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    images_dir = os.path.join("archive", "Images")
    if not os.path.isdir(images_dir):
        print(f"No images directory found at ` {images_dir} `")
        return

    img_pths = []
    for fname in os.listdir(images_dir):
        p = os.path.join(images_dir, fname)
        if os.path.isfile(p) and os.path.splitext(fname)[1].lower() in exts:
            img_pths.append(p)

    if not img_pths:
        print(f"No image files found in ` {images_dir} `")
        return

    captions_dict = load_captions_from_file("archive/captions.txt")

    positive_samples = prepare_samples(img_pths, captions_dict)
    samples = build_dataset(positive_samples)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    dataset = CrossModalDataset(samples, tokenizer, image_processor)

    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved {len(samples)} samples to `dataset.pkl`")


if __name__ == "__main__":
    main()