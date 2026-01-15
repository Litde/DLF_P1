import sys
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import importlib.util

# --- PATHS INSIDE CONTAINER ---
# These match the -v flags we will set in the Orchestrator
DATA_ROOT = "/app/data"  # Folder containing 'Images' folder and 'hidden_test.csv'
SUBMISSION_DIR = ""

# We assume the CSV is named 'hidden_test.csv' inside the data folder
TEST_CSV_PATH = os.path.join(DATA_ROOT, "hidden_test.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "Images")

# Default Transform (used if model doesn't provide custom one)
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_user_model():
    """Loads the user's model and optionally their custom transform.
    Returns: (model, device, transform)
    """
    model_path = os.path.join(SUBMISSION_DIR, "model.py")
    if not os.path.exists(model_path):
        raise FileNotFoundError("model.py not found in submission")

    # Dynamic Import
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)

    # Try to get custom transform from user's module
    if hasattr(user_module, 'get_transform'):
        try:
            custom_transform = user_module.get_transform()
            print("[INFO] Using custom transform from submission")
            transform = custom_transform
        except Exception as e:
            print(f"[WARN] Failed to load custom transform: {e}. Using default.")
            transform = default_transform
    else:
        print("[INFO] No custom transform found. Using default.")
        transform = default_transform

    # Load model
    model = user_module.SubmissionModel()

    weights_path = os.path.join(SUBMISSION_DIR, "weights.pth")
    if os.path.exists(weights_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device, transform
    else:
        raise FileNotFoundError("weights.pth not found")


def run_evaluation():
    try:
        model, device, transform = load_user_model()
        df = pd.read_csv(TEST_CSV_PATH)

        correct = 0
        total = 0

        for _, row in df.iterrows():
            img_filename = row['image']
            caption = row['caption']
            target = int(row['label'])

            img_path = os.path.join(IMAGES_DIR, img_filename)

            try:
                # Load & Preprocess
                image = Image.open(img_path).convert('RGB')
                img_tensor = transform(image).to(device)

                # --- INFERENCE ---
                # We expect a float between 0.0 and 1.0
                score = model.predict(img_tensor, caption)

                # Binarize prediction (Threshold 0.5)
                pred_label = 1 if score >= 0.5 else 0

                if pred_label == target:
                    correct += 1
                total += 1

            except Exception as e:
                # If a specific image fails, we skip but log it
                print(f"[WARN] Failed on {img_filename}: {e}")

        if total == 0:
            print("FINAL_SCORE:0.0")
        else:
            accuracy = correct / total
            print(f"FINAL_SCORE:{accuracy:.4f}")

    except Exception as e:
        print(f"FATAL_ERROR:{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run_evaluation()