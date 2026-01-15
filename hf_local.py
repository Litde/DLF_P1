"""Utilities for using HuggingFace models without internet access.

Usage (run once WITH internet):

    from hf_local import download_hf_models
    download_hf_models()  # saves into ./local_hf/

After that, `train.py` / models can load with `local_files_only=True`.

This file intentionally avoids argparse/CLI to keep it simple.
"""

from __future__ import annotations

import os
from pathlib import Path

from transformers import BertModel, BertTokenizer, ViTImageProcessor, ViTModel


VIT_ID = "google/vit-base-patch16-224-in21k"
BERT_ID = "bert-base-uncased"

DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "local_hf"
DEFAULT_VIT_DIRNAME = "vit-base"
DEFAULT_BERT_DIRNAME = "bert-base"


def get_vit_local_dir(local_dir: str | os.PathLike = DEFAULT_LOCAL_DIR) -> Path:
    return Path(local_dir) / DEFAULT_VIT_DIRNAME


def get_bert_local_dir(local_dir: str | os.PathLike = DEFAULT_LOCAL_DIR) -> Path:
    return Path(local_dir) / DEFAULT_BERT_DIRNAME


def download_hf_models(
    local_dir: str | os.PathLike = DEFAULT_LOCAL_DIR,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Download and save required HF artifacts into `local_dir`.

    Call this ONCE when internet is available.

    Args:
        local_dir: Target directory where models will be saved.
        force: If True, download and overwrite even if directories exist.

    Returns:
        Dict with paths: {"vit": "...", "bert": "..."}
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    vit_dir = get_vit_local_dir(local_dir)
    bert_dir = get_bert_local_dir(local_dir)

    # Vision
    if force and vit_dir.exists():
        # remove contents lazily by recreating
        for p in vit_dir.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
    vit_dir.mkdir(parents=True, exist_ok=True)
    ViTModel.from_pretrained(VIT_ID).save_pretrained(vit_dir)
    ViTImageProcessor.from_pretrained(VIT_ID).save_pretrained(vit_dir)

    # Text
    if force and bert_dir.exists():
        for p in bert_dir.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
    bert_dir.mkdir(parents=True, exist_ok=True)
    BertModel.from_pretrained(BERT_ID).save_pretrained(bert_dir)
    BertTokenizer.from_pretrained(BERT_ID).save_pretrained(bert_dir)

    return {"vit": str(vit_dir), "bert": str(bert_dir)}
