import pickle
from pathlib import Path
import numpy as np


def load_model(model_path: str):
    """Load trained model from pickle file."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_model(model, model_path: str):
    """Save trained model to pickle file."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def validate_features(features: list, expected_length: int = 4) -> np.ndarray:
    """Validate and convert input features to numpy array."""
    if len(features) != expected_length:
        raise ValueError(f"Expected {expected_length} features, got {len(features)}")
    return np.array(features).reshape(1, -1)
