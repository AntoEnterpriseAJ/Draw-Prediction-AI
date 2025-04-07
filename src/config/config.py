from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = MODEL_DIR / "data"
MODEL_PATH = MODEL_DIR / "mnist_model.pth"
