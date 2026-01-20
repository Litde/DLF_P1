from torch.utils.data import Dataset
from PIL import Image
import torch
from config import Config
import random
import pickle
from transformers import BertTokenizer
from transformers import ViTImageProcessor
import os
import re
import csv
from collections import defaultdict

# provide a tqdm progress bar if available, otherwise a noop passthrough
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

NEGATIVE_SUBS = {
    # People
    r"\bman\b": "woman",
    r"\bwoman\b": "man",
    r"\bboy\b": "girl",
    r"\bgirl\b": "boy",
    r"\bchild\b": "adult",
    r"\badult\b": "child",

    # Animals
    r"\bdog\b": "cat",
    r"\bdogs\b": "cats",
    r"\bhorse\b": "dog",
    r"\bbird\b": "dog",

    # Actions
    r"\brunning\b": "sleeping",
    r"\bjumping\b": "sitting",
    r"\bsitting\b": "standing",
    r"\bstanding\b": "lying",
    r"\bplaying\b": "resting",
    r"\bclimbing\b": "descending",
    r"\bwalking\b": "running",
    r"\bthrowing\b": "catching",
    r"\bcatching\b": "dropping",

    # Locations
    r"\bbeach\b": "street",
    r"\bstreet\b": "beach",
    r"\bpark\b": "room",
    r"\bwater\b": "sand",
    r"\bsnow\b": "grass",
    r"\bgrass\b": "snow",
    r"\bfield\b": "building",

    # Objects
    r"\bball\b": "stick",
    r"\bfrisbee\b": "ball",
    r"\bbike\b": "skateboard",
    r"\bskateboard\b": "bike",
    r"\bcamera\b": "toy",
    r"\brope\b": "chain",

    # Attributes
    r"\bred\b": "blue",
    r"\bblue\b": "red",
    r"\bblack\b": "white",
    r"\bwhite\b": "black",
    r"\bsmall\b": "large",
    r"\blarge\b": "small",

    # Numbers
    r"\bone\b": "two",
    r"\btwo\b": "three",
    r"\bthree\b": "two",
}

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


def generate_single_negative(caption: str, max_changes: int = 3) -> str:
    patterns = list(NEGATIVE_SUBS.items())
    random.shuffle(patterns)

    new_caption = caption
    changes = 0

    for pattern, replacement in patterns:
        if re.search(pattern, new_caption, flags=re.IGNORECASE):
            new_caption = re.sub(pattern, replacement, new_caption, flags=re.IGNORECASE)
            changes += 1
        if changes >= random.randint(1, max_changes):
            break

    if changes == 0:
        words = new_caption.split()
        if len(words) > 1:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
            new_caption = " ".join(words)
        else:
            new_caption += " different"

    return new_caption


def generate_negative_captions(
    caption: str,
    num_negatives: int = 5,
    max_attempts: int = 20
) -> list[str]:
    negatives = set()
    attempts = 0

    while len(negatives) < num_negatives and attempts < max_attempts:
        neg = generate_single_negative(caption)
        if neg != caption:
            negatives.add(neg)
        attempts += 1

    return list(negatives)


def build_dataset(positive_samples):
    """
    Accepts either:
    - a list of dicts with keys: 'image_path' (or 'path'/'img') and 'caption' (or 'text')
    - a list of image file paths (strings). In that case caption is derived from filename.
    Returns a shuffled list of samples with positive and corresponding negative examples.
    """
    samples = []
    for s in tqdm(positive_samples, desc="Building dataset", unit="sample"):
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

        neg_captions = generate_negative_captions(caption)

        for neg_caption in neg_captions:
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
    for path in tqdm(image_paths, desc="Preparing samples", unit="image"):
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
            for row in tqdm(reader, desc="Loading captions", unit="lines"):
                if len(row) >= 2:
                    image_name = os.path.splitext(row[0].strip())[0]
                    caption = row[1].strip()
                    captions_dict[image_name].append(caption)
    return dict(captions_dict)


def save_dataset(dataset, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)


def create_and_save_dataset(
    images_dir: str = os.path.join("archive", "Images"),
    captions_file: str = os.path.join("archive", "captions.txt"),
    out_filepath: str = "dataset.pkl",
    tokenizer_name: str = "bert-base-uncased",
    image_processor_name: str = "google/vit-base-patch16-224-in21k",
):
    """Builds a CrossModalDataset from images and captions, saves it using save_dataset(),
    and returns a tuple (dataset, samples, out_filepath).

    This reuses the existing helper functions in this module.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No images directory found at `{images_dir}`")

    img_pths = []
    for fname in tqdm(os.listdir(images_dir), desc="Scanning image files", unit="files"):
        p = os.path.join(images_dir, fname)
        if os.path.isfile(p) and os.path.splitext(fname)[1].lower() in exts:
            img_pths.append(p)

    if not img_pths:
        raise FileNotFoundError(f"No image files found in `{images_dir}`")

    captions_dict = load_captions_from_file(captions_file)

    positive_samples = prepare_samples(img_pths, captions_dict)
    samples = build_dataset(positive_samples)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    image_processor = ViTImageProcessor.from_pretrained(image_processor_name)

    dataset = CrossModalDataset(samples, tokenizer, image_processor)

    save_dataset(dataset, out_filepath)

    return dataset, samples, out_filepath


def load_true_false_csv(filepath, images_dir):
    """
    Load CSV with rows: image,caption,is_true(1 or 0)
    Returns a list of dicts: {'image_path': <abs path>, 'text': <caption>, 'label': 0|1}
    - images_dir is used to resolve filenames if the 'image' column contains only a basename.
    """
    samples = []
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = {os.path.basename(f): os.path.join(images_dir, f) for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in exts} if os.path.isdir(images_dir) else {}

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc="Loading true/false CSV", unit="lines"):
            if not row or len(row) < 3:
                continue
            img_col = row[0].strip()
            caption = row[1].strip()
            label_raw = row[2].strip()

            try:
                label = int(label_raw)
                label = 1 if label == 1 else 0
            except Exception:
                # skip malformed label
                continue

            # Try direct path first
            candidate = img_col
            if not os.path.isabs(candidate):
                candidate = os.path.join(images_dir, img_col)
            image_path = candidate if os.path.isfile(candidate) else None

            # Fallback: lookup by basename in images_dir
            if image_path is None:
                base = os.path.basename(img_col)
                if base in image_files:
                    image_path = image_files[base]
                else:
                    # try matching by id (without extension)
                    img_id = os.path.splitext(base)[0]
                    for fname in image_files:
                        if os.path.splitext(fname)[0] == img_id:
                            image_path = image_files[fname]
                            break

            if image_path is None or not os.path.isfile(image_path):
                # skip missing images
                continue

            samples.append({
                "image_path": image_path,
                "text": caption,
                "label": label
            })
    return samples


def create_and_save_dataset_v2(
    csv_path: str = "dataset_true_false_improved.csv",
    images_dir: str = os.path.join("archive", "Images"),
    out_filepath: str = "named_datasetV2.pkl",
    tokenizer_name: str = "bert-base-uncased",
    image_processor_name: str = "google/vit-base-patch16-224-in21k",
):
    """
    Build CrossModalDataset directly from a CSV (image,caption,is_true)
    and save it to `out_filepath` (default named_datasetV2.pkl).
    Returns: (dataset, samples, out_filepath)
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No images directory found at `{images_dir}`")

    samples = load_true_false_csv(csv_path, images_dir)

    if not samples:
        raise FileNotFoundError(f"No valid samples parsed from `{csv_path}`")

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    image_processor = ViTImageProcessor.from_pretrained(image_processor_name)

    dataset = CrossModalDataset(samples, tokenizer, image_processor)
    save_dataset(dataset, out_filepath)
    return dataset, samples, out_filepath


def main():
    try:
        dataset, samples, out_filepath = create_and_save_dataset()
        print(f"Saved {len(samples)} samples to `{out_filepath}`")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Failed to create dataset: {e}")


if __name__ == "__main__":
    # main()
    create_and_save_dataset_v2()